### ALL REQUIRED PYTHON MODULES.
import numpy as np
from utils.mesh import *
from utils.interface import *
###-----------------------------------------------------------------------------------
### FILE NAME:      poisson_scc.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    A simple 3D poisson equation test case for 3D IIM solver.
### NOTED:          A few more simply-constructed test case may be added later,
###                 and this file itself may be renamed.
###-----------------------------------------------------------------------------------


class poisson_scc(object):
    """ A simple 3D poisson equation (sin*cos*cos).

    Attributes:
        mesh        (mesh_uniform):     An uniform mesh in 3D cartesian coordinates 
                                        indicating the computational area.
        interface   (interface object): An interface built on same mesh object indicating
                                        where the jumps [u] and [u_n] occur.
        u_exact     (3D-array):         The exact solution of u.
        f_exact     (3D-array):         The exact(un-corrected) right hand sides.

    """

    def __init__(self, interface, mesh):
        """ Initialization of class 'poisson_scc'
            'u_exact' and 'f_exact' are computed.

        Args:
            mesh        (mesh_uniform):     An uniform mesh in 3D cartesian coordinates 
                                            indicating the computational area.
            interface   (interface object): An interface built on same mesh object indicating
                                            where the jumps [u] and [u_n] occur.
        Returns:
            None

        """
        self.mesh = mesh
        self.u_exact = np.ndarray(shape=(mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.f_exact = np.ndarray(shape=(mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)

        self.interface = interface
        
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    self.u_exact[i, j, k] = self.__u_exact(i, j, k)
                    self.f_exact[i, j, k] = self.__f_exact(i, j, k)
        return
    

    """ 'u_exact' and 'f_exact' on one point are computed.

    Args:
        i, j, k     (integer):      A Triplet indicating the point (x_i, y_j, z_k).

    Returns:
        u(x, y, z) = cos(x)*sin(y)*sin(z) if outsides the interface, 0 otherwise.  
        f(x, y, z) = -3*cos(x)*sin(y)*sin(z) if outsides the interface, 0 otherwise.
    """
    def __u_exact(self, i, j, k):
        if (self.interface.phi[i, j, k] >= 0):
            return np.cos(self.mesh.xs[i]) * np.sin(self.mesh.ys[j]) * np.sin(self.mesh.zs[k])
        return 0.0

    def __f_exact(self, i, j, k): 
        if (self.interface.phi[i, j, k] >= 0):
            return -3.0 * np.cos(self.mesh.xs[i]) * np.sin(self.mesh.ys[j]) * np.sin(self.mesh.zs[k])
        return 0.0
    

    """ All jump conditions on one point are computed.

    Args:
        x, y, z     (real):     A Triplet indicating the point (x, y, z) in cartesian coords.
    
    Note: 
        All points pass into the functions are assumed to be locating on the interface.

    Returns:
        Jump conditions (from neg. to pos.) [u], [u_x], [u_y], [u_z], [u_n], [f] on the interface.    
    """
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


""" MODULE TESTS
mesh = mesh_uniform()
inte = interface_ellipsoid(0.5, 0.5, 0.2, mesh)
a = poisson_scc(inte, mesh)
"""

### MODIFY HISTORY---
### 08.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------