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

class gmres(object):
    def __init__(self, r, max_iter=50):
        # Default x_0 = 0, r_0 = b.
        self.n = r.shape[0]
        self.max_iter = max_iter

        self.Q = np.zeros(shape=(self.n, max_iter), dtype=np.float64)
        self.H = np.zeros(shape=(max_iter + 1, max_iter), dtype=np.float64)
        self.k = 0

        self.s = np.zeros(shape=(max_iter + 1, 1), dtype=np.float64)
        self.s[0] = np.linalg.norm(r)
        
        self.Q[:, 0] = r.reshape(-1) / self.s[0]
        
        self.x = np.zeros(shape=(self.n, 1), dtype=np.float64)
        self.q = self.Q[:, 0]
        self.r = r

    def iterate(self, Akv, eps=1e-10):
        if (self.k >= self.max_iter - 1):
            return self.x, self.r, self.q

        t = Akv
        for i in range(self.k + 1):
            self.H[i, self.k] = np.dot(self.Q[:, i], t)
            t = t - self.H[i, self.k] * self.Q[:, i].reshape(-1, 1)
        
        self.H[self.k + 1, self.k] = np.linalg.norm(t)
        if (self.H[self.k + 1, self.k] > eps):
            self.Q[:, self.k + 1] = t.reshape(-1) / self.H[self.k + 1, self.k]
        
        #y = np.linalg.lstsq(self.H[:self.k + 2, :self.k + 1], self.s[:self.k + 2], rcond=None)[0].reshape(-1, 1)
        #x = np.dot(self.Q[:, :self.k + 1], y).reshape(-1, 1)
        y = np.linalg.lstsq(self.H, self.s, rcond=None)[0].reshape(-1, 1)
        self.x = np.dot(self.Q, y).reshape(-1)

        p = self.s.reshape(-1, 1) - np.dot(self.H, y)
        self.r = np.dot(self.Q, p[:-1, :])

        self.q = self.Q[:, self.k + 1]

        self.k = self.k + 1
        #if (self.H[self.k + 1, self.k] <= eps):
        #    self.k = self.max_iter
        return self.x, self.r, self.q

mesh = mesh_uniform(multiplier=1)
inte = interface_ellipsoid_aug(0.6, 0.5, np.sqrt(2.0)/4.0, mesh)


pde = helmholtz_scc_aug0(inte, mesh, lambda_c=-5)
pde.set_jump_u_n(pde.irr_jump_u_nT[1:])

#"""
solu = helmholtz_IIM_solver(pde)
u = solu.u
c = solu.irr_corr

for i in range(mesh.n_x + 1):
    for j in range(mesh.n_y + 1):
        for k in range(mesh.n_z + 1):
            if (inte.phi[i, j, k] > 0):
                # 2 Types.
                if (inte.irr[i, j, k] > 0):
            #u[i, j, k] = pde.u_exact[i, j, k]
                    u[i, j, k] = u[i, j, k] - c[inte.irr[i, j, k]] #+-
                if (inte.irr[i, j, k] < 0):
                    #print(inte.irr[i, j, k], c[np.abs(inte.irr[i, j, k])])
                    #u[i, j, k] = u[i, j, k] - c[np.abs(inte.irr[i, j, k])] #+-
                    pass

b_GT = pde.irr_jump_u_B
b_IP = pde.get_jump_u_b(u)
#print(b_GT.shape, b_IP.shape)

for i in range(mesh.n_x + 1):
    for j in range(mesh.n_y + 1):
        for k in range(mesh.n_z + 1):
            if (inte.irr[i, j, k] > 0):
                err = b_GT[inte.irr[i, j, k] - 1] - b_IP[inte.irr[i, j, k] - 1]
                if (np.abs(err) > 1e-3):
                    print("ERR: ", err, "SIGN: ", inte.phi[i, j, k])

print('dis: ', mesh.h_x)

#plt.plot(b_GT)
plt.plot(b_IP-b_GT)
#plt.plot(c)

#print(np.max(np.abs(u - pde.u_exact)))
print(np.max(np.abs(b_GT - b_IP)))

plt.show()
#"""

"""
def A(q):
    pde.set_jump_u_n(q)

    solu = helmholtz_IIM_solver(pde)
    u = solu.u
    c = solu.irr_corr

    for i in range(mesh.n_x + 1):
        for j in range(mesh.n_y + 1):
            for k in range(mesh.n_z + 1):
                if (inte.irr[i, j, k] > 0 and inte.phi[i, j, k] > 0):
                    u[i, j, k] = u[i, j, k] - c[inte.irr[i, j, k]] #+-
    return pde.get_jump_u_b(u)
    

b = pde.irr_jump_u_B
x = np.zeros(shape=(inte.n_irr, 1), dtype=np.float64)

r = b - A(x)


gm = gmres(r, 305)
q = r / np.linalg.norm(r)

for i in range(300):
    print('itr:', i)
    Aq = A(q).reshape(-1, 1)

    x, r, q = gm.iterate(Aq)
    #print(x, '<- X')
    #print(r.reshape(-1), '<- ESTI R')

    #r = b - np.dot(A, x)
    #print(b.reshape(-1) - A(x), '<- TRUE R')

    #input()
    #print(x)
"""




### MODIFY HISTORY---
### 22.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------