import numpy as np

""" DEBUG
from mesh import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""

class interface_ellipsoid(object):
    def __init__(self, a, b, c, mesh):
        self.a = a
        self.b = b
        self.c = c
        self.phi = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.irr = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.int)

        self.phi_x = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_y = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_z = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_xx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_xy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_xz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_yx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_yy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_yz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_zx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_zy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_zz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    self.phi[i, j, k] = self.__phi(mesh.xs[i], mesh.ys[j], mesh.zs[k])

        # Irregular points are sought AFTER phi is calculated.
        self.n_irr = 0
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    if (i == 0 or i == mesh.n_x or
                        j == 0 or j == mesh.n_y or
                        k == 0 or k == mesh.n_z):
                        continue

                    # Irregular points.
                    if (self.phi[i, j, k] == 0.0):
                        self.irr[i, j, k] = 1
                    else:
                        # A little trick.
                        dirs = [0, 0, 1]
                        for l in range(3):
                            dirs =  np.roll(dirs, 1)
                            if ((self.phi[i, j, k] <= 0.0 and self.phi[i + dirs[0], j + dirs[1], k + dirs[2]] > 0.0) or
                                (self.phi[i, j, k] > 0.0 and self.phi[i + dirs[0], j + dirs[1], k + dirs[2]] <= 0.0)):
                                self.irr[i, j, k] = 1
                            if ((self.phi[i, j, k] <= 0.0 and self.phi[i - dirs[0], j - dirs[1], k - dirs[2]] > 0.0) or
                                (self.phi[i, j, k] > 0.0 and self.phi[i - dirs[0], j - dirs[1], k - dirs[2]] <= 0.0)):
                                self.irr[i, j, k] = 1

                    if (self.irr[i, j, k] == 1):
                        self.n_irr = self.n_irr + 1
                        self.irr[i, j, k] = self.n_irr

                    # Derivatives.
                    self.phi_x[i, j, k] = self.__phi_x(i, j, k, mesh.h_x)
                    self.phi_y[i, j, k] = self.__phi_y(i, j, k, mesh.h_y)
                    self.phi_z[i, j, k] = self.__phi_z(i, j, k, mesh.h_z)

                    self.phi_xx[i, j, k] = self.__phi_xx(i, j, k, mesh.h_x)
                    self.phi_xy[i, j, k] = self.__phi_xy(i, j, k, mesh.h_x, mesh.h_y)
                    self.phi_xz[i, j, k] = self.__phi_xz(i, j, k, mesh.h_x, mesh.h_z)
                    
                    self.phi_yx[i, j, k] = self.__phi_yx(i, j, k, mesh.h_x, mesh.h_y)
                    self.phi_yy[i, j, k] = self.__phi_yy(i, j, k, mesh.h_y)
                    self.phi_yz[i, j, k] = self.__phi_yz(i, j, k, mesh.h_y, mesh.h_z)
                    
                    self.phi_zx[i, j, k] = self.__phi_zx(i, j, k, mesh.h_x, mesh.h_z)
                    self.phi_zy[i, j, k] = self.__phi_zy(i, j, k, mesh.h_y, mesh.h_z)
                    self.phi_zz[i, j, k] = self.__phi_zz(i, j, k, mesh.h_z)

    # Level set function (INACCURATE, iterative method for SDF can be used instead).
    def __phi(self, x, y, z):
        return x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2 - 1.0

    # Level set function derivatives (public). 
    def __phi_x(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] - self.phi[i - 1, j, k]) / (2.0 * h_x)
    def __phi_y(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] - self.phi[i, j - 1, k]) / (2.0 * h_y)
    def __phi_z(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] - self.phi[i, j, k - 1]) / (2.0 * h_z)
    # 2nd-order derivatives.
    def __phi_xx(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] + self.phi[i - 1, j, k] - 2.0 * self.phi[i, j, k]) / (h_x * h_x)
    def __phi_xy(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def __phi_xz(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    
    def __phi_yx(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def __phi_yy(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] + self.phi[i, j - 1, k] - 2.0 * self.phi[i, j, k]) / (h_y * h_y)
    def __phi_yz(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    
    def __phi_zx(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    def __phi_zy(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    def __phi_zz(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] + self.phi[i, j, k - 1] - 2.0 * self.phi[i, j, k]) / (h_z * h_z)
    
    # Level set function derivatives (public). 
    def phi_x_(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] - self.phi[i - 1, j, k]) / (2.0 * h_x)
    def phi_y_(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] - self.phi[i, j - 1, k]) / (2.0 * h_y)
    def phi_z_(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] - self.phi[i, j, k - 1]) / (2.0 * h_z)
    # 2nd-order derivatives.
    def phi_xx_(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] + self.phi[i - 1, j, k] - 2.0 * self.phi[i, j, k]) / (h_x * h_x)
    def phi_xy_(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def phi_xz_(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    
    def phi_yx_(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def phi_yy_(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] + self.phi[i, j - 1, k] - 2.0 * self.phi[i, j, k]) / (h_y * h_y)
    def phi_yz_(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    
    def phi_zx_(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    def phi_zy_(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    def phi_zz_(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] + self.phi[i, j, k - 1] - 2.0 * self.phi[i, j, k]) / (h_z * h_z)
    
    
""" DEBUG
mesh = mesh_uniform(multiplier=2)
inte = interface_ellipsoid(0.5, 0.5, 0.2, mesh)

fig = plt.figure()            
ax = fig.gca(projection='3d')

x = []
y = []
z = []
for i in range(mesh.n_x):
    for j in range(mesh.n_y):
        for k in range(mesh.n_z):
            if (inte.irr[i, j, k] > 0):
                x.append(mesh.xs[i])
                y.append(mesh.ys[j])
                z.append(mesh.zs[k])
ax.scatter(x, y, z, linewidth=1)
plt.show()
"""