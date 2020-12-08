import numpy as np
import functools

def root_p2(a, b, c, eps=1e-10):
    # NO ROOT EXISTENCE CHECKING.
    r1 = np.inf
    r2 = np.inf
    
    # Degenerate to p1.
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

""" DEBUG
a = 1
b = 4
c = 7

[r1, r2] = root_p2(a, b, c)
print (r1, r2)
"""

def linear_interp_3d(ratio_x, ratio_y, ratio_z, vals):
    # All loops unrolled.
    res = 0
    res = res + vals[0, 0, 0] * (1 - ratio_x) * (1 - ratio_y) * (1 - ratio_z)
    res = res + vals[1, 0, 0] * (ratio_x) * (1 - ratio_y) * (1 - ratio_z)
    res = res + vals[0, 1, 0] * (1 - ratio_x) * (ratio_y) * (1 - ratio_z)
    res = res + vals[0, 0, 1] * (1 - ratio_x) * (1 - ratio_y) * (ratio_z)
    res = res + vals[1, 1, 0] * (ratio_x) * (ratio_y) * (1 - ratio_z)
    res = res + vals[1, 0, 1] * (ratio_x) * (1 - ratio_y) * (ratio_z)
    res = res + vals[0, 1, 1] * (1 - ratio_x) * (ratio_y) * (ratio_z)
    res = res + vals[1, 1, 1] * (ratio_x) * (ratio_y) * (ratio_z)
    
    return res

# N-point Lagrange interpolation. (Numerical unstable)
def Pn_interp_1d(x, xs, offset, fs, n):
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

# 3D N-point Lagrange interpolation. (Isotropic)
def Pn_interp_3d(x, y, z, xs, ys, zs, offset_i, offset_j, offset_k, fs, n):
    a = np.ndarray(shape=(n, ), dtype=np.float64)
    b = np.ndarray(shape=(n, ), dtype=np.float64)
    c = np.ndarray(shape=(n, ), dtype=np.float64)
    
    # Xtra attention on the looping order.
    for k in range(n):
        for j in range(n):
            for i in range(n):
                a[i] = fs[i + offset_i, j + offset_j, k + offset_k]
            b[j] = Pn_interp_1d(x, xs, offset_i, a, n)
        c[k] = Pn_interp_1d(y, ys, offset_j, b, n)
    return Pn_interp_1d(z, zs, offset_k, c, n)
