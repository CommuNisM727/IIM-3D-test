### ALL REQUIRED PYTHON MODULES.
import numpy as np

from poisson_IIM_solver import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
###-----------------------------------------------------------------------------------
### FILE NAME:      gmres_test.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    A GMRES test.
### NOTED:          Might be deleted
###-----------------------------------------------------------------------------------

"""
def gmres(A, b, x_0, max_iter, eps=1e-10):
    n = x_0.shape[0]
    r = b - np.dot(A, x_0)

    Q = np.zeros(shape=(n, max_iter), dtype=np.float64)
    X = np.zeros(shape=(n, max_iter), dtype=np.float64)
    H = np.zeros(shape=(max_iter + 1, max_iter), dtype=np.float64)
    
    Q[:, 0] = r / np.linalg.norm(r)
    s = np.zeros(max_iter + 1)
    s[0] = np.linalg.norm(r)

    for k in range(max_iter - 1):
        print('itr: ', k)

        t = A * Q[:, k].reshape(-1, 1)# * 5
        print(t, '<- new')
        
        # Orthogonalization.
        for i in range(k + 1):
            H[i, k] = np.dot(Q[:, i], t)
            t = t - H[i, k] * Q[:, i].reshape(-1, 1)
        
        H[k + 1, k] = np.linalg.norm(t)
        if (H[k + 1, k] > eps):
            Q[:, k + 1] = t.reshape(-1) / H[k + 1, k]
        

        y = np.linalg.lstsq(H, s, rcond=None)[0].reshape(-1, 1)
        x = np.dot(Q, y).reshape(-1)
        
        print(np.dot(A, x) - b, '<- R')
        q = s.reshape(-1, 1) - np.dot(H, y)
        r0 = -np.dot(Q, q[:-1, :])
        print(r0.reshape(-1), '<- R_esti')
        
        X[:, k] = x# + x_0

        if (H[k + 1, k] <= eps):
            break

    print("RESULT: ")
    #print(H, '<- H')
    print(Q, '<- Q')
    print(X, '<- X')

A = np.matrix('1 1; 0 1')
b = np.array([3, 4])
x = np.array([0, 0])

gmres(A, b, x, 6)
"""

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

"""
A = np.matrix('1 1; 0 1') # (-1, 4)
b = np.array([3, 4]).reshape(-1, 1)
x = np.array([0, 0]).reshape(-1, 1)

r = b - np.dot(A, x)
q = r / np.linalg.norm(r)
print('R0', r)

gm = gmres(r, 20)
for i in range(15):
    print('itr:', i)
    Aq = np.dot(A, q).reshape(-1, 1)
    print('new:', Aq)
    x, r, q = gm.iterate(Aq)
    print(x, '<- X')
    print(r.reshape(-1), '<- ESTI R')

    #r = b - np.dot(A, x)
    print(b.reshape(-1) - np.dot(A, x), '<- TRUE R')

    input()
    #print(x)
"""

### MODIFY HISTORY---
### 21.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------