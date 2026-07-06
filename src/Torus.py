import math
from src.Malha import Malha
from src.Ponto import Ponto
from utils.MeshReader.ObjReader import FaceData
from utils.Scene.sceneSchema import MaterialData, ColorData


class Torus:
    # material fixo do toro — não vem do JSON da cena
    _MATERIAL_PADRAO = MaterialData(
        name="toro_dourado",
        color=ColorData(0.83, 0.68, 0.21),
        ks=ColorData(0.9, 0.9, 0.9),
        ka=ColorData(0.83, 0.68, 0.21),
        kr=ColorData(0.0, 0.0, 0.0),
        kt=ColorData(0.0, 0.0, 0.0),
        ns=40,
        ni=1.0,
        d=1.0,
    )

    def __init__(self, R, r, transforms: list,material: MaterialData = None, n_u=16, n_v=12):
        self.R = R
        self.r = r
        self.n_u = n_u
        self.n_v = n_v
        self.material = material 

        vertices = self._gerar_vertices()
        faces = self._gerar_faces()

        self.malha = Malha.from_dados(vertices, faces, self.material, transforms)

    def _gerar_vertices(self):
        vertices = []
        for i in range(self.n_u):
            u = 2 * math.pi * i / self.n_u
            for j in range(self.n_v):
                v = 2 * math.pi * j / self.n_v
                x = (self.R + self.r * math.cos(v)) * math.cos(u)
                y = self.r * math.sin(v)
                z = (self.R + self.r * math.cos(v)) * math.sin(u)
                vertices.append(Ponto(x, y, z))
        return vertices

    def _indice(self, i, j):
        i = i % self.n_u
        j = j % self.n_v
        return i * self.n_v + j

    def _gerar_faces(self):
        faces = []
        for i in range(self.n_u):
            for j in range(self.n_v):
                a = self._indice(i, j)
                b = self._indice(i + 1, j)
                c = self._indice(i + 1, j + 1)
                d = self._indice(i, j + 1)
                faces.append(FaceData(vertice_indice=[a, b, c]))
                faces.append(FaceData(vertice_indice=[a, c, d]))
        return faces

    def intersectar(self, raio):
        return self.malha.intersectar(raio)