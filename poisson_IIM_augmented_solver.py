### ALL REQUIRED PYTHON MODULES.
import numpy as np

from poisson_IIM_solver import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
###-----------------------------------------------------------------------------------
### FILE NAME:      poisson_IIM_augmented_solver.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    A 3D IIM augmented solver.
### NOTED:          Might be split into multiple modules.
###-----------------------------------------------------------------------------------

"""
def gmres(A, b, x_0, max_iter):
    n = x_0.shape[0]
    r = b - np.dot(A, x_0)

    Q = np.zeros(shape=(n, max_iter), dtype=np.float64)
    X = np.zeros(shape=(n, max_iter), dtype=np.float64)
    H = np.zeros(shape=(max_iter + 1, max_iter), dtype=np.float64)
    
    Q[:, 0] = r / np.linalg.norm(r)
    for k in range(0, max_iter):
        t = A * Q[:, k].reshape(-1, 1)

        # Orthogonalization.
        for i in range(k):
            H[i, k] = np.dot(Q[:, i], t)
            t = t - H[i, k] * Q[:, i].reshape(-1, 1)
        
        H[k + 1, k] = np.linalg.norm(t)
        if (H[k + 1, k] != 0 and k != max_iter - 1):
            Q[:, k + 1] = t.reshape(-1) / H[k + 1, k]
        
        s = np.zeros(max_iter + 1)
        s[0] = np.linalg.norm(r)
        y = np.linalg.lstsq(H, s, rcond=None)[0].reshape(-1, 1)
        
        X[:, k] = np.dot(Q, y).reshape(-1) + x_0


A = np.matrix('1 1; 0 1')
b = np.array([3, 4])
x = np.array([0, 0])

gmres(A, b, x, 50)
"""


class gmres(object):
    def __init__(self, r, max_iter=50):
        # Default x_0 = 0, r_0 = b.
        self.n = r.shape[0]
        self.max_iter = max_iter

        self.Q = np.zeros(shape=(self.n, max_iter), dtype=np.float64)
        self.H = np.zeros(shape=(max_iter + 1, max_iter), dtype=np.float64)
        self.k = 0

        self.s = np.zeros(self.max_iter + 1)
        self.s[0] = np.linalg.norm(r)
        
        self.Q[:, 0] = r.reshape(-1) / self.s[0]

    def iterate(self, Akr):
        if (self.k >= self.max_iter - 1):
            return
        
        t = Akr
        # Orthogonalization.
        for i in range(self.k):
            self.H[i, self.k] = np.dot(self.Q[:, i], t)
            t = t - self.H[i, self.k] * self.Q[:, i].reshape(-1, 1)
            
        self.H[self.k + 1, self.k] = np.linalg.norm(t)
        if (self.H[self.k + 1, self.k] != 0):
            self.Q[:, self.k + 1] = t.reshape(-1) / self.H[self.k + 1, self.k]
        
        self.k = self.k + 1

        y = np.linalg.lstsq(self.H, self.s, rcond=None)[0].reshape(-1, 1)
        return np.dot(self.Q, y).reshape(-1, 1)
        
A = np.matrix('1 1; 0 1')
b = np.array([3, 4]).reshape(-1, 1)
x = np.array([0, 0]).reshape(-1, 1)

r = b - np.dot(A, x)
print(r)
gm = gmres(r)
for i in range(30):
    Ax = np.dot(A, x)
    x = gm.iterate(Ax)
    #print(x)

### MODIFY HISTORY---
### 21.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------