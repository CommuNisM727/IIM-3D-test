### ALL REQUIRED PYTHON MODULES.
import numpy as np
""" DEBUG
from mesh import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""
###-----------------------------------------------------------------------------------
### FILE NAME:      interface.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    Indicating the interface through level-set function \phi.
### NOTED:          A simple ellipsoid interface and its inaccurate level-set 
###                 function is given.
###                 More interface may be added later.
###-----------------------------------------------------------------------------------


class interface_ellipsoid(object):
    """ A simple ellipsoid interface (x^2/a^2 + y^2/b^2 + z^2/c^2 = 1).

    Attributes:
        a, b, c     (real):             Elliptic radius along x, y, z.
        irr         (3D-array):         Indices of irregular mesh points [1, n_irr].
        n_irr       (integer):          Number of irregular mesh points.
        phi         (3D-array):         Level-set function \phi computed on mesh points.
        phi_        (3D-array):         1st-order derivatives of \phi. 
        phi__       (3D-array):         2nd-order derivatives of \phi.

    """
    def __init__(self, a, b, c, mesh):
        """ Initialization of class 'interface_ellipsoid'
            Irrgular mesh points 'irr' and 'n_irr' are computed.
            Level-set function '\phi' and its derivatives '\phi_' are computed.

        Args:
            a, b, c     (real):             Elliptic radius along eigen-direction x, y, z.
            mesh        (mesh_uniform):     An uniform mesh in 3D cartesian coordinates 
                                            indicating the computational area.

        Returns:
            None

        """

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
        
        # STEP 1:  Find the value of level-set function \phi on all mesh points.
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    self.phi[i, j, k] = self.__phi(mesh.xs[i], mesh.ys[j], mesh.zs[k])

        # STEP 2:  Find the derivatives of \phi and map irregular points to a index.
        self.n_irr = 0
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    if (i == 0 or i == mesh.n_x or
                        j == 0 or j == mesh.n_y or
                        k == 0 or k == mesh.n_z):
                        continue

                    # Irregular points must be sought AFTER \phi is calculated.
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

                    # Derivatives of \phi.
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


    def __phi(self, x, y, z):
        """ Level-set function \phi on one point are computed.
            This implementation is an INACCURATE one based on the analytical equation of 
            ellipsoid, more accurate result can be obtained if a Signed-Distance-Function
            is calculated through iterative methods or so.

        Args:
            x, y, z     (real):     A Triplet indicating the point (x, y, z).

        Returns:
            \phi(x, y, z) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1.
        """
        return x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2 - 1.0

    """ All derivatives (central difference) of level-set function \phi on one 
        point are computed, private method called by the initialization function.

    Args:
        i, j, k         (integer):      A Triplet indicating the point (x_i, y_j, z_k).
        h_x, h_y, h_z   (real):         Distance between points along x, y, z axis.

    Returns:
        Central difference approximation of all derivatives of \phi.
    """
    # 1st-order.
    def __phi_x(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] - self.phi[i - 1, j, k]) / (2.0 * h_x)
    def __phi_y(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] - self.phi[i, j - 1, k]) / (2.0 * h_y)
    def __phi_z(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] - self.phi[i, j, k - 1]) / (2.0 * h_z)
    # 2nd-order.
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


""" MODULE TESTS
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

### MODIFY HISTORY---
### 09.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------