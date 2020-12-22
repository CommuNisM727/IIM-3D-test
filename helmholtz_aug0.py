### ALL REQUIRED PYTHON MODULES.
import time
import numpy as np

from helmholtz_IIM_solver import *
from helmholtz_scc_aug0 import *
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

mesh = mesh_uniform(multiplier=2)
inte = interface_ellipsoid(0.6, 0.5, np.sqrt(2.0)/4.0, mesh)


jump_u_n = np.zeros(shape=(inte.n_irr, 1), dtype=np.float64)
a = helmholtz_scc_aug0(inte, mesh, jump_u_n, lambda_c=5)
a.set_jump_u_n(a.irr_jump_u_nT[1:])


scc = helmholtz_IIM_solver(a)


#"""

### MODIFY HISTORY---
### 22.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------