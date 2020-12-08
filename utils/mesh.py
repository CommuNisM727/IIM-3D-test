### ALL REQUIRED PYTHON MODULES.
import numpy as np
###-----------------------------------------------------------------------------------
### FILE NAME:      mesh.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    A simple uniform mesh generator.
### NOTED:          Different types of mesh may be added.
###-----------------------------------------------------------------------------------


class mesh_uniform(object):
    """ A simple 3D poisson equation (sin*cos*cos).

    Attributes:
        x_inf, x_sup    (real):     The lower and upper bound for x.
        y_inf, y_sup    (real):     The lower and upper bound for y.
        z_inf, z_sup    (real):     The lower and upper bound for z.
        multiplier      (integer):  1D mesh size = 16 * multiplier.
        n_x, n_y, n_z   (integer):  Number of points along x, y, z axis.
        h_x, h_y, h_z   (real):     Distance between points along x, y, z axis.
        xs, ys, zs      (1D-array): Cartesian coords of points.

    """
    def __init__(self, x_inf=-1.0, x_sup=1.0, y_inf=-1.0, y_sup=1.0, z_inf=-1.0, z_sup=1.0, multiplier=1):
        self.x_inf = x_inf
        self.x_sup = x_sup
        self.y_inf = y_inf
        self.y_sup = y_sup
        self.z_inf = z_inf
        self.z_sup = z_sup

        self.multiplier = multiplier
                
        self.n_x = 16*self.multiplier
        self.n_y = 16*self.multiplier
        self.n_z = 16*self.multiplier
        self.h_x = (x_sup - x_inf) / self.n_x
        self.h_y = (y_sup - y_inf) / self.n_y
        self.h_z = (z_sup - z_inf) / self.n_z
        self.xs = np.linspace(start=x_inf, stop=x_sup, num=self.n_x + 1, endpoint=True, dtype=np.float64)
        self.ys = np.linspace(start=y_inf, stop=y_sup, num=self.n_y + 1, endpoint=True, dtype=np.float64)
        self.zs = np.linspace(start=z_inf, stop=z_sup, num=self.n_z + 1, endpoint=True, dtype=np.float64)


### MODIFY HISTORY---
### 08.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------