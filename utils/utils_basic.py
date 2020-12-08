### ALL REQUIRED PYTHON MODULES.
import numpy as np
###-----------------------------------------------------------------------------------
### FILE NAME:      utils_basic.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    Some basic math utilities.
### NOTED:          None.
###-----------------------------------------------------------------------------------


def root_p2(a, b, c, eps=1e-10):
    """ A module for computing the roots of quadratic equation a*x^2 + b*x + c.
        No root-existency checks are done here.

    Args:
        a, b, c     (real):             Coefficients of equation.
        eps         (real, optional):   Coefficients smaller than eps will be regarded as 0.

    Returns:
        [r1, r2]:   Two roots of the quadratic equation.

    """
    
    r1 = np.inf
    r2 = np.inf
    
    # Degenerate to P_1.
    if (np.abs(a) <= eps and np.abs(b) > eps):
        r1 = -c / b
        r2 = r1
        return [r1, r2]
    # Non-degenerate case.
    det = b*b - 4*a*c
    if (det >= 0):
        det = np.sqrt(det)
        if (b >= 0):
            r1 = (-b - det) / (2.0 * a)
        else:
            r1 = (-b + det) / (2.0 * a)
        if (det == 0):
            r2 = r1
        else:
            r2 = c / (a * r1)
    return [r1, r2]

""" MODULE TESTS
a = 1
b = 4
c = 7

[r1, r2] = root_p2(a, b, c)
print (r1, r2)
"""

def Pn_interp_1d(x, xs, offset, fs, n):
    """ A module for computing f(x) through Lagrange interpolating from sampled points.
        'n' shoule be small for numerical stability.

    Args:
        x       (real):     The cartesian coord of point to be interpolated.
        xs      (1D-array): The cartesian coords of ALL possible sampling points.
        offset  (integer):  The offset of initial index of sampling point (x_[i+offset]).
        fs      (1D-array): The interpolating values of REQUIRED sampling points.
        n       (integer):  The number of interpolating points.

    Returns:
        res:    Interpolating result f(x).

    """

    res = 0.0
    for i in range(n):
        nume = 1.0
        deno = 1.0
        for j in range(n):
            if (i == j):
                continue
            nume = nume * (x - xs[j + offset])
            deno = deno * (xs[i + offset] - xs[j + offset])
        res = res + fs[i] * nume/deno
    return res


def Pn_interp_3d(x, y, z, xs, ys, zs, offset_i, offset_j, offset_k, fs, n):
    """ A module for computing f(x, y, z) through 3D Lagrange interpolating 
        from N^3 sampled points, 'n' shoule be small for numerical stability.

    Args:
        x, y, z     (real):     The cartesian coords of point to be interpolated.
        xs, ys, zs  (1D-array): The cartesian coords of ALL possible sampling points.
        offset_     (integer):  The offset in x, y, z of initial index of sampling point.
        fs          (1D-array): The interpolating values of ALL sampling points.
        n           (integer):  The number of interpolating points in one direction.

    Returns:
        Interpolating result f(x, y, z).

    """
    a = np.ndarray(shape=(n, ), dtype=np.float64)
    b = np.ndarray(shape=(n, ), dtype=np.float64)
    c = np.ndarray(shape=(n, ), dtype=np.float64)
    
    # Xtra attention to the looping order.
    for k in range(n):
        for j in range(n):
            for i in range(n):
                a[i] = fs[i + offset_i, j + offset_j, k + offset_k]
            b[j] = Pn_interp_1d(x, xs, offset_i, a, n)
        c[k] = Pn_interp_1d(y, ys, offset_j, b, n)
    return Pn_interp_1d(z, zs, offset_k, c, n)


### MODIFY HISTORY---
### 09.12.2020      FILE CREATED.           ---727
###-----------------------------------------------------------------------