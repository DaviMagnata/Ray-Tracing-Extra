import math
import sys
from src.Torus import Torus
from src.Ponto import Ponto
from src.Vetor import Vetor
from src.Raio import Raio
from src.Esfera import Esfera
from src.Plano import Plano
from src.Malha import Malha
from utils.Scene.sceneParser import SceneJsonLoader
from utils.Scene.sceneSchema import TransformData
import sys
# Pequeno deslocamento usado para afastar raios secundários da superfície de or igem
# e evitar auto-interseção causada por imprecisão de ponto flutuante.
SHADOW_EPS = 1e-4

# Profundidade máxima da recursão de reflexão/refração (entrega 4). Cada raio
# secundário (refletido ou refratado) pode gerar novos raios secundários; sem
# esse limite, duas superfícies espelhadas frente a frente recursionariam para
# sempre. 5 é suficiente para reflexões e travessias de vidro convincentes.
MAX_DEPTH = 5


def base_camera(data):
    """Constrói a base ortonormal da câmera: três eixos perpendiculares entre si.

    w  aponta para TRÁS (de M em direção a C) — eixo de profundidade.
    u  aponta para a DIREITA — calculado como w × up.
    v  aponta para CIMA     — calculado como u × w.

    A ordem w × up (e não up × w) é necessária porque a cena usa a convenção
    de que +X no mundo equivale à direita da imagem. Com up × w o eixo u ficaria
    apontando para -X e a imagem sairia espelhada horizontalmente.

    Os vetores são normalizados para formar uma base ortonormal: comprimento 1
    garante que o parâmetro t das interseções corresponda à distância real."""
    C    = data.camera.lookfrom       # posição da câmera no mundo
    M    = data.camera.lookat         # ponto para onde a câmera aponta
    v_up = data.camera.up_vector      # vetor "para cima" do mundo (geralmente Y)

    w = (C - M).normalizar()                  # eixo de profundidade (aponta para trás)
    u = w.prodVetorial(v_up).normalizar()     # eixo horizontal (aponta para a direita)
    v = u.prodVetorial(w)                     # eixo vertical (aponta para cima)

    return C, u, v, w


def gerar_raio(i, j, C, u, v, w, largura, altura, d):
    """Gera o raio que passa pelo centro do pixel (i, j).

    Usa o modelo de câmera pinhole: a tela fica a d unidades à frente
    da câmera (C - w*d) e cada pixel é mapeado para coordenadas NDC.

    NDC (Normalized Device Coordinates): índices de pixel são convertidos
    para o intervalo [-aspect, +aspect] no eixo horizontal e [-1, +1] no
    vertical, com (0, 0) no centro exato da imagem.

    O +0.5 nos índices faz o raio passar pelo centro do pixel, não pelo canto.
    Multiplicar px pela razão de aspecto evita que objetos circulares apareçam
    como elipses em imagens não quadradas."""
    aspect = largura / altura

    px = (2 * (i + 0.5) / largura  - 1) * aspect   # coordenada horizontal NDC
    py =  1 - 2 * (j + 0.5) / altura               # coordenada vertical NDC (j=0 é o topo)

    # O ponto na tela é C - w*d + u*px + v*py. O vetor direção é esse ponto menos C,
    # o que cancela C e deixa só -w*d + u*px + v*py.
    direcao = ((-w) * d + u * px + v * py).normalizar()

    return Raio(C, direcao)


def criar_objetos(scene):
    """Converte os dados carregados do JSON em objetos de cena concretos.
    Itera sobre scene.objects e instancia Esfera, Plano ou Malha conforme o tipo.

    Para malhas (type == "mesh"), o material vem do arquivo .mtl referenciado
    pelo .obj — não do JSON da cena. O material do JSON é passado como fallback
    para faces que não declaram material via 'usemtl'."""
    objetos = []
    for obj in scene.objects:
        if obj.obj_type == "sphere":
            centro = obj.get_ponto("center")
            raio   = obj.get_num("radius")
            objetos.append(Esfera(centro, raio, obj.material))

        elif obj.obj_type == "plane":
            ponto  = obj.get_ponto("point_on_plane")
            normal = obj.get_vetor("normal")
            objetos.append(Plano(ponto, normal, obj.material))

        elif obj.obj_type == "mesh":
            path = obj.get_property("path")
            objetos.append(Malha(path, obj.material, obj.transforms))

        elif obj.obj_type == "torus":
            R   = obj.get_num("R")
            r   = obj.get_num("r")
            n_u = int(obj.numeric_data.get("n_u", 30))
            n_v = int(obj.numeric_data.get("n_v", 20))
            
            objetos.append(Torus(R=R, r=r, n_u=n_u, n_v=n_v,material=obj.material, transforms=obj.transforms))

    return objetos


def encontrar_hit_mais_proximo(raio: Raio, objetos):
    """Testa o raio contra todos os objetos e retorna o HitInfo de menor t.
    Retorna None se nenhum objeto for atingido."""
    melhor = None
    for obj in objetos:
        hit = obj.intersectar(raio)
        if hit is not None and (melhor is None or hit.t < melhor.t):
            melhor = hit
    return melhor


def em_sombra(P: Ponto, N: Vetor, luz_pos: Ponto, objetos) -> bool:
    """Verifica se o ponto P está em sombra em relação à luz.

    Lança um raio de sombra de P na direção da luz. Se algum objeto bloqueia
    com t em (0, |luz - P|), o ponto está em sombra. Desloca a origem do
    raio levemente ao longo da normal para evitar auto-interseção."""
    direcao_luz = luz_pos - P                       # Ponto - Ponto = Vetor
    distancia_luz = direcao_luz.modulo()
    if distancia_luz < 1e-9:
        return False
    L = direcao_luz / distancia_luz                 # normalizado

    origem = P + N * SHADOW_EPS
    raio_sombra = Raio(origem, L)

    for obj in objetos:
        hit = obj.intersectar(raio_sombra)
        if hit is not None and SHADOW_EPS < hit.t < distancia_luz:
            return True
    return False


def phong(hit, raio, scene, objetos):
    """Calcula a cor em um ponto de interseção pelo modelo de iluminação de Phong:

        I = ka·Ia + Σ [ kd·(L·N)·IL + ks·(R·V)^η·IL ]

    - termo ambiente:  ka * Ia (luz ambiente global)
    - para cada luz visível (não bloqueada por sombra):
        - difuso:    kd * max(0, L·N) * IL
        - especular: ks * max(0, R·V)^η * IL
    Esta função calcula apenas a iluminação LOCAL. Os termos de reflexão e
    refração (kr·Ir, kt·It) são somados por tracar() (entrega 4), que chama phong
    como o termo de base de cada nível da recursão.

    Vetores (todos normalizados):
        N — normal da superfície em P (vem do HitInfo)
        L — de P até a posição da luz
        R — reflexão de -L em torno de N:  R = 2(L·N)N - L
        V — de P até o observador (para raios primários, oposto à direção do raio)
    """
    P   = hit.point
    N   = hit.normal
    mat = hit.material

    Ia = scene.global_light.color

    # Termo ambiente
    r = mat.ka.r * Ia.r
    g = mat.ka.g * Ia.g
    b = mat.ka.b * Ia.b

    # Vetor de visualização: para o observador (oposto à direção do raio incidente)
    V = -raio.direcao

    for luz in scene.light_list:
        if em_sombra(P, N, luz.pos, objetos):
            continue

        direcao_luz   = luz.pos - P
        distancia_luz = direcao_luz.modulo()
        if distancia_luz < 1e-9:
            continue
        L = direcao_luz / distancia_luz

        ln = L.prodEscalar(N)
        if ln <= 0:
            # Luz atrás da superfície — não contribui (nem difuso nem especular)
            continue

        # Difuso
        r += mat.color.r * ln * luz.color.r
        g += mat.color.g * ln * luz.color.g
        b += mat.color.b * ln * luz.color.b

        # Especular
        R = N * (2.0 * ln) - L
        rv = R.prodEscalar(V)
        if rv > 0 and mat.ns > 0:
            spec = rv ** mat.ns
            r += mat.ks.r * spec * luz.color.r
            g += mat.ks.g * spec * luz.color.g
            b += mat.ks.b * spec * luz.color.b

    return r, g, b


def _tem_cor(c) -> bool:
    """True se o coeficiente de cor (kr ou kt) tem alguma componente não nula —
    ou seja, vale a pena lançar o raio secundário correspondente."""
    return c.r != 0.0 or c.g != 0.0 or c.b != 0.0


def refletir(D: Vetor, N: Vetor) -> Vetor:
    """Reflete a direção incidente D em torno da normal N (ambas normalizadas).

    Fórmula clássica de reflexão especular:  R = D - 2*(D·N)*N

    Como N vem do HitInfo já orientada contra o raio (D·N < 0), o raio refletido
    R sai da superfície pelo mesmo lado de onde o raio chegou."""
    return D - N * (2.0 * D.prodEscalar(N))


def refratar(D: Vetor, N: Vetor, n1: float, n2: float):
    """Direção do raio refratado pela lei de Snell, ou None em reflexão interna total.

    D  — direção incidente (normalizada)
    N  — normal orientada contra D (D·N < 0)
    n1 — índice de refração do meio de onde o raio vem
    n2 — índice de refração do meio para onde o raio entra

    Snell em forma vetorial:  T = n*D + (n*cosθi - cosθt)*N,  com n = n1/n2.
    Se n²·(1 - cos²θi) > 1, o ângulo crítico foi ultrapassado → reflexão interna
    total (não há raio transmitido) → retorna None."""
    n = n1 / n2
    cos_i = -D.prodEscalar(N)               # > 0, pois N está contra D
    sin2_t = n * n * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None                         # reflexão interna total
    cos_t = math.sqrt(1.0 - sin2_t)
    return D * n + N * (n * cos_i - cos_t)


def tracar(raio: Raio, scene, objetos, profundidade: int):
    """Ray tracing recursivo: cor que o 'raio' enxerga na cena.

    1. Acha o objeto mais próximo. Se não houver, devolve o fundo (preto).
    2. Calcula a cor local pelo modelo de Phong (ambiente + difuso + especular).
    3. Se ainda há orçamento de recursão e o material é reflexivo (kr > 0),
       lança um raio refletido e soma  kr · I_r.
    4. Se o material é transparente (kt > 0), lança um raio refratado (lei de
       Snell, usando material.ni) e soma  kt · I_t.

    Isso completa a equação da entrega 3 com os dois termos que faltavam:
        I = Phong  +  k_r · I_r  +  k_t · I_t
    """
    hit = encontrar_hit_mais_proximo(raio, objetos)
    if hit is None:
        return (0.0, 0.0, 0.0)

    r, g, b = phong(hit, raio, scene, objetos)

    # Parou no limite de recursão: devolve só a cor local.
    if profundidade >= MAX_DEPTH:
        return r, g, b

    mat = hit.material
    N   = hit.normal
    D   = raio.direcao

    # --- Reflexão: k_r · I_r ---
    if _tem_cor(mat.kr):
        R = refletir(D, N).normalizar()
        origem = hit.point + N * SHADOW_EPS       # desloca para fora da superfície
        rr, rg, rb = tracar(Raio(origem, R), scene, objetos, profundidade + 1)
        r += mat.kr.r * rr
        g += mat.kr.g * rg
        b += mat.kr.b * rb

    # --- Refração: k_t · I_t ---
    if _tem_cor(mat.kt):
        # front_face decide o sentido da travessia: ar→material na entrada,
        # material→ar na saída. Ar tem índice 1.0.
        if hit.front_face:
            n1, n2 = 1.0, mat.ni
        else:
            n1, n2 = mat.ni, 1.0

        T = refratar(D, N, n1, n2)
        if T is not None:                          # None = reflexão interna total
            T = T.normalizar()
            origem = hit.point - N * SHADOW_EPS     # desloca para dentro da superfície
            tr, tg, tb = tracar(Raio(origem, T), scene, objetos, profundidade + 1)
            r += mat.kt.r * tr
            g += mat.kt.g * tg
            b += mat.kt.b * tb

    return r, g, b


def _to_byte(c: float) -> int:
    """Converte cor float em [0, ∞) para byte [0, 255], aplicando clamp."""
    if c <= 0.0:
        return 0
    if c >= 1.0:
        return 255
    return int(c * 255)


def renderizar(scene_path="utils/input/sampleScene.json"):
    """Renderiza a cena e imprime o resultado no stdout no formato PPM.

    Para cada pixel (i, j) lança um raio primário e chama tracar(), que faz o
    ray tracing recursivo: iluminação de Phong com sombras no ponto atingido,
    mais os raios secundários de reflexão (kr) e refração (kt) até MAX_DEPTH.
    Se nenhum objeto for atingido, o pixel fica preto (fundo).

    O PPM é impresso no stdout — redirecionar para arquivo gera o arquivo de imagem."""
    scene   = SceneJsonLoader.load_file(scene_path)
    largura = scene.camera.image_width
    altura  = scene.camera.image_height
    d       = scene.camera.screen_distance

    C, u, v, w = base_camera(scene)
    objetos     = criar_objetos(scene)


    # Cabeçalho PPM: tipo P3 (texto RGB), dimensões e valor máximo por canal
    linhas = [f"P3\n{largura} {altura}\n255"]

    print(f"Renderizando {largura}x{altura} ({len(objetos)} objetos, "
          f"{len(scene.light_list)} luzes)...", file=sys.stderr)

    for j in range(altura):
        if j % 10 == 0 or j == altura - 1:
            pct = (j + 1) * 100 // altura
            print(f"  linha {j + 1}/{altura} ({pct}%)", file=sys.stderr)
        for i in range(largura):
            raio = gerar_raio(i, j, C, u, v, w, largura, altura, d)
            # Raio primário com profundidade 0; tracar() cuida da recursão de
            # reflexão/refração e devolve a cor já somada (Phong + k_r·I_r + k_t·I_t).
            r, g, b = tracar(raio, scene, objetos, 0)
            linhas.append(f"{_to_byte(r)} {_to_byte(g)} {_to_byte(b)}")

    sys.stdout.write("\n".join(linhas) + "\n")


if __name__ == "__main__":
    scene_file = sys.argv[1] if len(sys.argv) > 1 else "utils/input/sampleScene.json"
    renderizar(scene_file)
