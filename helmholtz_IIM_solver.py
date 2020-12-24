### ALL REQUIRED PYTHON MODULES.
import time
import numpy as np
import utils.helmholtz3D

from helmholtz_scc import *
from utils.utils_basic import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
###-----------------------------------------------------------------------------------
### FILE NAME:      helmholtz_IIM_solver.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    A 3D IIM solver.
### NOTED:          Might be split into multiple modules.
###-----------------------------------------------------------------------------------


class helmholtz_IIM_solver(object):
    """ A simple 3D IIM helmholtz solver.
    Attributes:
        pde         (pde object):       A poisson or helmholtz equation.

        rhs_corr    (1D-array):         Right-hand side correction terms on irregular points.
        irr_corr    (1D-array):         Correction terms on irregular points.
        u           (3D-array):         Numerical solution to the equation.
        error       (double):           Numerical error to the ground-truth. 
        
    """

    def __init__(self, pde):
        """ Initialization of class 'helmholtz IIM solver'.
            Initialization, solving, and error estimation are all done here.
        Args:
            pde         (pde object):       The PDE to be solved.
        Returns:
            None

        """

        self.pde = pde
        n_irr = self.pde.interface.n_irr
        if(hasattr(pde.interface, 'n_app')):
            n_irr = n_irr + pde.interface.n_app
            print('AUGMENTED POINT: ', pde.interface.n_app)

        self.rhs_corr = np.zeros(shape=(n_irr + 1, ), dtype=np.float64)
        self.irr_corr = np.zeros(shape=(n_irr + 1, ), dtype=np.float64)
        self.u = np.asfortranarray(np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_y + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F'))
        self.error = 0.0

        time_a=time.time()
        for i in range(1, self.pde.mesh.n_x):
            for j in range(1, self.pde.mesh.n_y):
                for k in range(1, self.pde.mesh.n_z):
                    if (self.pde.interface.irr[i, j, k] != 0):
                        index = self.pde.interface.irr[i, j, k]
                        self.__irregular_projection_corr(index, i, j, k)

        self.__solve()
        time_b=time.time()
        print('T_solve: ', time_b-time_a)

        self.__error_estimate()

    def __irregular_projection_corr(self, index, i, j, k):
        """ A module for computing corrections.

        Args:
            i, j, k     (integer):      The index of coords of irregular point.

        Returns:
            None

        Computation:
            correction term [u] + d*[u_n] + 1/2 d^2*[u_{nn}].

        """

        abs_index = np.abs(index)
        d = self.pde.interface.irr_dist[abs_index]
        corr = self.pde.irr_jump_u[abs_index]   \
        + d * self.pde.irr_jump_u_n[abs_index]  \
        + 0.5*d*d * self.pde.irr_jump_u_nn[abs_index]

        self.irr_corr[abs_index] = corr
        if (index < 0):
            corr = self.pde.irr_jump_u[abs_index]   \
            + d * self.pde.irr_jump_u_n[abs_index]
            
            print(index, corr)
            return

        # x-.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i - 1, j, k] > 0):
            index_ = self.pde.interface.irr[i - 1, j, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] + corr / self.pde.mesh.h_x**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i - 1, j, k] <= 0):
            index_ = self.pde.interface.irr[i - 1, j, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] - corr / self.pde.mesh.h_x**2
        # x+.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i + 1, j, k] > 0):
            index_ = self.pde.interface.irr[i + 1, j, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] + corr / self.pde.mesh.h_x**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i + 1, j, k] <= 0):
            index_ = self.pde.interface.irr[i + 1, j, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] - corr / self.pde.mesh.h_x**2
        
        # y-.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j - 1, k] > 0):
            index_ = self.pde.interface.irr[i, j - 1, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] + corr / self.pde.mesh.h_y**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j - 1, k] <= 0):
            index_ = self.pde.interface.irr[i, j - 1, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] - corr / self.pde.mesh.h_y**2
        # y+.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j + 1, k] > 0):
            index_ = self.pde.interface.irr[i, j + 1, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] + corr / self.pde.mesh.h_y**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j + 1, k] <= 0):
            index_ = self.pde.interface.irr[i, j + 1, k]
            self.rhs_corr[index_] = self.rhs_corr[index_] - corr / self.pde.mesh.h_y**2
        
        # z-.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j, k - 1] > 0):
            index_ = self.pde.interface.irr[i, j, k - 1]
            self.rhs_corr[index_] = self.rhs_corr[index_] + corr / self.pde.mesh.h_z**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j, k - 1] <= 0):
            index_ = self.pde.interface.irr[i, j, k - 1]
            self.rhs_corr[index_] = self.rhs_corr[index_] - corr / self.pde.mesh.h_z**2
        # z+.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j, k + 1] > 0):
            index_ = self.pde.interface.irr[i, j, k + 1]
            self.rhs_corr[index_] = self.rhs_corr[index_] + corr / self.pde.mesh.h_z**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j, k + 1] <= 0):
            index_ = self.pde.interface.irr[i, j, k + 1]
            self.rhs_corr[index_] = self.rhs_corr[index_] - corr / self.pde.mesh.h_z**2
        
    def __solve(self):
        for i in range(self.pde.mesh.n_x + 1):
            for j in range(self.pde.mesh.n_y + 1):
                for k in range(self.pde.mesh.n_z + 1):
                    # RHS initialization & modification.
                    self.u[i, j, k] = self.pde.f_exact[i, j, k]
                    
                    if (self.pde.interface.irr[i, j, k] > 0):
                        self.u[i, j, k] = self.u[i, j, k] - self.rhs_corr[self.pde.interface.irr[i, j, k]]
                    
                    # Boundary conditions.
                    if (i == 0 or i == self.pde.mesh.n_x or
                        j == 0 or j == self.pde.mesh.n_y or
                        k == 0 or k == self.pde.mesh.n_z):
                            self.u[i, j, k] = self.pde.u_exact[i, j, k]


        # Dummy arrays.
        BDXS = np.zeros(shape=(self.pde.mesh.n_y + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDXF = np.zeros(shape=(self.pde.mesh.n_y + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDYS = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDYF = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDZS = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_y + 1), dtype=np.float64, order='F')
        BDZF = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_y + 1), dtype=np.float64, order='F')
        PERTRB = np.array(0, dtype=np.float64)
        IERROR = np.array(-1, dtype=np.int32)
        W = np.zeros(shape=(
            30 + self.pde.mesh.n_x + self.pde.mesh.n_y + 5*self.pde.mesh.n_z    \
            + np.max([self.pde.mesh.n_x, self.pde.mesh.n_y, self.pde.mesh.n_z]) \
            + 7*((self.pde.mesh.n_x + 1)//2 + (self.pde.mesh.n_y + 1)//2)), dtype=np.float64)
    
        #print("FORTRAN OUTPUT:")
        #"""
        utils.helmholtz3D.hw3crtt(
            xs=np.array(self.pde.mesh.x_inf, dtype=np.float64), xf=np.array(self.pde.mesh.x_sup, dtype=np.float64), l=np.array(self.pde.mesh.n_x, dtype=np.int32), lbdcnd=np.array(1, dtype=np.int32), bdxs=BDXS, bdxf=BDXF,
            ys=np.array(self.pde.mesh.y_inf, dtype=np.float64), yf=np.array(self.pde.mesh.y_sup, dtype=np.float64), m=np.array(self.pde.mesh.n_y, dtype=np.int32), mbdcnd=np.array(1, dtype=np.int32), bdys=BDYS, bdyf=BDYF,
            zs=np.array(self.pde.mesh.z_inf, dtype=np.float64), zf=np.array(self.pde.mesh.z_sup, dtype=np.float64), n=np.array(self.pde.mesh.n_z, dtype=np.int32), nbdcnd=np.array(1, dtype=np.int32), bdzs=BDZS, bdzf=BDZF,
            elmbda=np.array(self.pde.lambda_c, dtype=np.float64), f=self.u, pertrb=PERTRB, ierror=IERROR, w=W)
        #"""
        #print("P, E:", PERTRB, IERROR)

        """ Error plot.
        N = self.pde.mesh.n_x//2
        res = self.u[N, :, :] - self.pde.u_exact[N, :, :]
        Y, Z = np.meshgrid(self.pde.mesh.ys, self.pde.mesh.zs)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        #surf = ax.plot_surface(Y, Z, self.pde.u_exact[N, :, :], cmap=cm.coolwarm,
        #               linewidth=0, antialiased=False)

        surf = ax.plot_surface(Y, Z, res, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        """

    def __error_estimate(self, dis_select=True, dis_multiplier=1.5):
        for i in range(self.pde.mesh.n_x + 1):
            for j in range(self.pde.mesh.n_y + 1):
                for k in range(self.pde.mesh.n_z + 1):
                    if (dis_select or np.abs(self.pde.interface.phi[i, j, k]) <= dis_multiplier*self.pde.mesh.h_x):
                        self.error = np.max([self.error, np.abs(self.u[i, j, k] - self.pde.u_exact[i, j, k])])

        print("MAX ERROR:", self.error)

""" MODULE TEST
mesh = mesh_uniform(multiplier=2)
inte = interface_ellipsoid(0.6, 0.5, np.sqrt(2.0)/4.0, mesh)
a = helmholtz_scc(inte, mesh, lambda_c=-5)
scc = helmholtz_IIM_solver(a)
"""

### MODIFY HISTORY---
### 22.12.2020      FILE CREATED.           ---727
### 22.12.2020      Ver. 0.2 CREATED.       ---727
###-----------------------------------------------------------------------