### ALL REQUIRED PYTHON MODULES.
import numpy as np
from utils.mesh import *
from utils.interface import *
###-----------------------------------------------------------------------------------
### FILE NAME:      helmholtz_scc_aug0.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    A 3D augmented helmholtz equation test case for 3D IIM solver.
### NOTED:          A few more simply-constructed test case may be added later,
###                 and this file itself may be renamed.
###-----------------------------------------------------------------------------------


class helmholtz_scc_aug0(object):
    """ A simple 3D helmholtz equation [\delta u + \lambda_c u = f] (u = sin*cos*cos).

    Attributes:
        mesh            (mesh_uniform):     An uniform mesh in 3D cartesian coordinates 
                                            indicating the computational area.
        interface       (interface object): An interface built on same mesh object indicating
                                            where the jumps [u] and [u_n] occur.
        lambda_c        (real):             The coefficient \lambda_c.

        u_exact         (3D-array):         The exact solution of u.
        f_exact         (3D-array):         The exact(un-corrected) right hand sides.
        u_bound         (1D-array):         The exact boundary projection points.

        irr_jump_u      (1D-array):         An array of [u].
        irr_jump_f      (1D-array):         An array of [f].
        irr_jump_u_n    (1D-array):         An array of [u_n].
        irr_jump_u_nn   (1D-array):         An array of [u_{nn}].

    """

    def __init__(self, interface, mesh, jump_u_n, lambda_c=1):
        """ Initialization of class 'helmholtz_scc_aug0'
            'u_exact' and 'f_exact' are computed.

        Args:
            mesh        (mesh_uniform):     An uniform mesh in 3D cartesian coordinates 
                                            indicating the computational area.
            interface   (interface object): An interface built on same mesh object indicating
                                            where the jumps [u] and [u_n] occur.
            lambda_c    (real):             The coefficient \lambda_c.

        Returns:
            None

        """

        self.mesh = mesh
        self.u_exact = np.ndarray(shape=(mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.f_exact = np.ndarray(shape=(mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)

        self.interface = interface
        self.lambda_c = lambda_c

        self.irr_jump_u = np.ndarray(shape=(interface.n_irr + 1, ), dtype=np.float64)
        self.irr_jump_f = np.ndarray(shape=(interface.n_irr + 1, ), dtype=np.float64)
        self.irr_jump_u_n = np.ndarray(shape=(interface.n_irr + 1, ), dtype=np.float64)
        self.irr_jump_u_nT = np.ndarray(shape=(interface.n_irr + 1, ), dtype=np.float64)
        self.irr_jump_u_nn = np.ndarray(shape=(interface.n_irr + 1, ), dtype=np.float64)

        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    self.u_exact[i, j, k] = self.__u_exact(i, j, k)
                    self.f_exact[i, j, k] = self.__f_exact(i, j, k)

                    if (self.interface.irr[i, j, k] > 0):
                        self.__irregular_projection_jump1(self.interface.irr[i, j, k])

        self.set_jump_u_n(jump_u_n)
        
        return

    def set_jump_u_n(self, jump_u_n):
        #self.irr_jump_u_n[0] = 0
        for i in range(1, self.interface.n_irr + 1):
            self.irr_jump_u_n[i] = jump_u_n[i - 1]
            #self.irr_jump_u_n[i] = self.irr_jump_u_nT[i]
            pass

        for i in range(self.mesh.n_x + 1):
            for j in range(self.mesh.n_y + 1):
                for k in range(self.mesh.n_z + 1):
                    if (self.interface.irr[i, j, k] > 0):
                        self.__irregular_projection_jump2(self.interface.irr[i, j, k], i, j, k)
        

    def get_jump_u_b(self, u_corr):
        jump_u_b = np.zeros(shape=(self.interface.n_irr + 1, 1), dtype=np.float64)

        for i in range(self.mesh.n_x + 1):
            for j in range(self.mesh.n_y + 1):
                for k in range(self.mesh.n_z + 1):
                    if (self.interface.irr[i, j, k] > 0):
                        index = self.interface.irr[i, j, k]
                        x = self.interface.irr_proj[index, 0]
                        y = self.interface.irr_proj[index, 1]
                        z = self.interface.irr_proj[index, 2]

                        i = int((x - self.mesh.x_inf) / self.mesh.h_x)
                        j = int((y - self.mesh.y_inf) / self.mesh.h_y)
                        k = int((z - self.mesh.z_inf) / self.mesh.h_z)

                        Pn_interp_3d_partial = lambda arr, n: Pn_interp_3d(x, y, z, 
                                                self.mesh.xs, self.mesh.ys, self.mesh.zs,
                                                i - int((n - 1) / 2), j - int((n - 1) / 2), k - int((n - 1) / 2),
                                                arr, n)
                        # Tricubic interpolating (4^3 points).
                        jump_u_b[index - 1] = Pn_interp_3d_partial(u_corr, 2)

                        pass

        return

    """ 'u_exact' and 'f_exact' on one point are computed.

    Args:
        i, j, k     (integer):      A Triplet indicating the point (x_i, y_j, z_k).

    Returns:
        u(x_i, y_j, z_k) = cos(x_i)*sin(y_j)*sin(z_k) if outsides the interface, 0 otherwise.  
        f(x_i, y_j, z_k) = (\lambda_c - 3)*cos(x_i)*sin(y_j)*sin(z_k) if outsides the interface, 0 otherwise.
    """
    def __u_exact(self, i, j, k):
        if (self.interface.phi[i, j, k] <= 0):
            return np.cos(self.mesh.xs[i]) * np.sin(self.mesh.ys[j]) * np.sin(self.mesh.zs[k])
        return 0.0

    def __f_exact(self, i, j, k): 
        if (self.interface.phi[i, j, k] <= 0):
            return (self.lambda_c - 3.0) * np.cos(self.mesh.xs[i]) * np.sin(self.mesh.ys[j]) * np.sin(self.mesh.zs[k])
        return 0.0
    
    def __irregular_projection_jump1(self, index):
        """ A module for computing and saving the basic information of irregular point projection on the interface.

        Args:
            index       (integer>0):    The index of irregular point.

        Returns:
            None

        """

        x = self.interface.irr_proj[index, 0]
        y = self.interface.irr_proj[index, 1]
        z = self.interface.irr_proj[index, 2]
        
        self.irr_jump_u[index] = self.jump_u(x, y, z)
        self.irr_jump_f[index] = self.jump_f(x, y, z)
        self.irr_jump_u_nT[index] = \
            self.jump_u_x(x, y, z) * self.interface.irr_Xi[index, 0] \
        +   self.jump_u_y(x, y, z) * self.interface.irr_Xi[index, 1] \
        +   self.jump_u_z(x, y, z) * self.interface.irr_Xi[index, 2]
        #self.irr_jump_u_n[index] = self.irr_jump_u_nT[index]
        return

    def __irregular_projection_jump2(self, index, i, j, k, norm_l1=3, norm_l2=2.4, n_points=16):
        """ A module for computing [u_{nn}] using least square.

        Args:
            i, j, k     (integer):      The index of coords of irregular point.
            norm_l1     (real):         Neighbour search |area|_1 < norm_l1 * h.
            norm_l2     (real):         Neighbour search |area|_2 < norm_l2 * h.
            n_points    (integer):      Number of neighbour points used.

        Returns:
            None

        Computation:
            [u_{nn}] are assigned here. ([u_{nn}] = [f_n] - \Kappa [u_n] - [u_{surface Laplacian}])

        """

        x = self.interface.irr_proj[index, 0]
        y = self.interface.irr_proj[index, 1]
        z = self.interface.irr_proj[index, 2]
        
        neighbours = np.ndarray(shape=((int(2*norm_l1))**3, ), dtype=np.int)
        distances = np.ndarray(shape=((int(2*norm_l1))**3, ), dtype=np.float64)
        
        n = 0
        for offset_i in range(-norm_l1, norm_l1 + 1):
            for offset_j in range(-norm_l1, norm_l1 + 1):
                for offset_k in range(-norm_l1, norm_l1 + 1):
                    # Neighbour point index.
                    i_ = i + offset_i
                    j_ = j + offset_j
                    k_ = k + offset_k

                    if (self.interface.irr[i_, j_, k_] > 0):
                        index_ = self.interface.irr[i_, j_, k_]
                        x_ = self.interface.irr_proj[index_, 0]
                        y_ = self.interface.irr_proj[index_, 1]
                        z_ = self.interface.irr_proj[index_, 2]

                        dist = np.sqrt((x-x_)**2 + (y-y_)**2 + (z-z_)**2)
                        if (dist <= norm_l2*self.mesh.h_x):
                            neighbours[n] = index_
                            distances[n] = dist
                            n = n + 1
        
        # Selecting the nearest 'n_points' neighbours.
        neighbours = neighbours[:n]
        distances = distances[:n]
        
        order = np.argsort(distances)
        neighbours = neighbours[order]
        neighbours = neighbours[:n_points]

        neighbour_jump_u = np.ndarray(shape=(n_points, ), dtype=np.float64)
        
        # n_features (fixed) * n_points.
        n_features = 15
        neighbour_dict = np.ndarray(shape=(n_points, n_features), dtype=np.float64)

        for n_ in range(n_points):
            index_ = neighbours[n_]
            dx = self.interface.irr_proj[index_, 0] - x
            dy = self.interface.irr_proj[index_, 1] - y
            dz = self.interface.irr_proj[index_, 2] - z
            
            neighbour_jump_u[n_] = self.irr_jump_u[index_] - self.irr_jump_u[index]
            Eta = self.interface.irr_Eta[index, 0] * dx \
                + self.interface.irr_Eta[index, 1] * dy \
                + self.interface.irr_Eta[index, 2] * dz
            Tau = self.interface.irr_Tau[index, 0] * dx \
                + self.interface.irr_Tau[index, 1] * dy \
                + self.interface.irr_Tau[index, 2] * dz

            # o0 (Not removed for future possible modification)
            neighbour_dict[n_, 0] = 0.0 #1.0
            # o1
            neighbour_dict[n_, 1] = Eta
            neighbour_dict[n_, 2] = Tau
            # o2
            neighbour_dict[n_, 3] = 0.5*Eta*Eta #
            neighbour_dict[n_, 4] = Eta*Tau
            neighbour_dict[n_, 5] = 0.5*Tau*Tau #
            # o3
            neighbour_dict[n_, 6] = Eta*Eta*Eta
            neighbour_dict[n_, 7] = Eta*Eta*Tau
            neighbour_dict[n_, 8] = Eta*Tau*Tau
            neighbour_dict[n_, 9] = Tau*Tau*Tau
            # o4
            neighbour_dict[n_, 10] = Eta*Eta*Eta*Eta
            neighbour_dict[n_, 11] = Eta*Eta*Eta*Tau
            neighbour_dict[n_, 12] = Eta*Eta*Tau*Tau
            neighbour_dict[n_, 13] = Eta*Tau*Tau*Tau
            neighbour_dict[n_, 14] = Tau*Tau*Tau*Tau
        
        derivs = np.linalg.lstsq(neighbour_dict, neighbour_jump_u, rcond=1e-10)[0]
        
        self.irr_jump_u_nn[index] = self.irr_jump_f[index]              \
        - self.lambda_c * self.irr_jump_u[index]                        \
        - self.interface.irr_Kappa[index] * self.irr_jump_u_n[index]    \
        - (derivs[3] + derivs[5])
        #self.irr_jump_u_nn[index] = self.jump_u_nn(x, y, z, self.interface.irr_Xi[index, 0], self.interface.irr_Xi[index, 1], self.interface.irr_Xi[index, 2])
        return

    """ All jump conditions on one point are computed.

    Args:
        x, y, z     (real):     A Triplet indicating the point (x, y, z) in cartesian coords.
    
    Note: 
        All points pass into the functions are assumed to be locating on the interface.

    Returns:
        Jump conditions (from neg. to pos.) [u], [u_x], [u_y], [u_z], [u_n], [f] on the interface.    
    """
    def jump_u(self, x, y, z):
        return -np.cos(x) * np.sin(y) * np.sin(z)

    def jump_u_x(self, x, y, z):
        return np.sin(x) * np.sin(y) * np.sin(z)
    def jump_u_y(self, x, y, z):
        return -np.cos(x) * np.cos(y) * np.sin(z)
    def jump_u_z(self, x, y, z):
        return -np.cos(x) * np.sin(y) * np.cos(z)
    def jump_u_n(self, x, y, z):
        norm = np.sqrt(x**2 / self.interface.a**4 + y**2 / self.interface.b**4 + z**2 / self.interface.c**4)
        normal_x = x/self.interface.a**2 / norm
        normal_y = y/self.interface.b**2 / norm
        normal_z = z/self.interface.c**2 / norm
        
        return self.jump_u_x(x, y, z)*normal_x + self.jump_u_y(x, y, z)*normal_y + self.jump_u_z(x, y, z)*normal_z

    def jump_u_nn(self, x, y, z, n_x, n_y, n_z):
        # For error checking purpose.
        u_xx = np.cos(x) * np.sin(y) * np.sin(z)
        u_xy = np.sin(x) * np.cos(y) * np.sin(z)
        u_xz = np.sin(x) * np.sin(y) * np.cos(z)
        u_yx = np.sin(x) * np.cos(y) * np.sin(z)
        u_yy = -np.cos(x) * np.sin(y) * np.sin(z)
        u_yz = -np.cos(x) * np.cos(y) * np.cos(z)
        u_zx = np.sin(x) * np.sin(y) * np.cos(z)
        u_zy = -np.cos(x) * np.cos(y) * np.cos(z)
        u_zz = np.cos(x) * np.sin(y) * np.sin(z)

        return n_x * (u_xx * n_x + u_xy * n_y + u_xz * n_z) \
        +      n_y * (u_yx * n_x + u_yy * n_y + u_yz * n_z) \
        +      n_z * (u_zx * n_x + u_zy * n_y + u_zz * n_z)

    def jump_f(self, x, y, z):
        return -(self.lambda_c - 3.0) * np.cos(x) * np.sin(y) * np.sin(z)


""" MODULE TESTS
mesh = mesh_uniform()
inte = interface_ellipsoid(0.5, 0.5, 0.2, mesh)
a = helmholtz_scc(inte, mesh)
"""

### MODIFY HISTORY---
### 22.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------