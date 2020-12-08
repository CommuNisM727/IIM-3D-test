import numpy as np

from utils.mesh import *
from utils.interface import *

class poisson_scc(object):
    def __init__(self, interface, mesh):
        self.mesh = mesh
        self.u_exact = np.ndarray(shape=(mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.f_exact = np.ndarray(shape=(mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)

        self.interface = interface
        
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    self.u_exact[i, j, k] = self.__u_exact(i, j, k)
                    self.f_exact[i, j, k] = self.__f_exact(i, j, k)

    # Poisson equation.
    def __u_exact(self, i, j, k):
        if (self.interface.phi[i, j, k] >= 0):
            return np.cos(self.mesh.xs[i]) * np.sin(self.mesh.ys[j]) * np.sin(self.mesh.zs[k])
        return 0.0

    def __f_exact(self, i, j, k): 
        if (self.interface.phi[i, j, k] >= 0):
            return -3.0 * np.cos(self.mesh.xs[i]) * np.sin(self.mesh.ys[j]) * np.sin(self.mesh.zs[k])
        return 0.0
    
    # Jump conditions (neg->pos).
    # Assume the point locates on the interface (drop all the checking).
    def jump_u(self, x, y, z):
        return np.cos(x) * np.sin(y) * np.sin(z)

    def jump_u_x(self, x, y, z):
        return -np.sin(x) * np.sin(y) * np.sin(z)
    def jump_u_y(self, x, y, z):
        return np.cos(x) * np.cos(y) * np.sin(z)
    def jump_u_z(self, x, y, z):
        return np.cos(x) * np.sin(y) * np.cos(z)
    def jump_u_n(self, x, y, z):
        norm = np.sqrt(x**2 / self.interface.a**4 + y**2 / self.interface.b**4 + z**2 / self.interface.c**4)
        normal_x = x/self.interface.a**2 / norm
        normal_y = y/self.interface.b**2 / norm
        normal_z = z/self.interface.c**2 / norm
        
        return self.jump_u_x(x, y, z)*normal_x + self.jump_u_y(x, y, z)*normal_y + self.jump_u_z(x, y, z)*normal_z

    def jump_f(self, x, y, z):
        return -3.0 * np.cos(x) * np.sin(y) * np.sin(z)


#mesh = mesh_uniform()
#inte = interface_ellipsoid(0.5, 0.5, 0.2, mesh)
#a = poisson_scc(inte, mesh)
