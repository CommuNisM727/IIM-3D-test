import numpy as np

# Parameter configuration for Poisson equation.
x_inf = -1.0
y_inf = -1.0
z_inf = -1.0
x_sup = 1.0
y_sup = 1.0
z_sup = 1.0

multiplier = 4
n_x = 8*multiplier
n_y = 8*multiplier
n_z = 8*multiplier

h_x = (x_sup - x_inf) / n_x
h_y = (y_sup - y_inf) / n_y
h_z = (z_sup - z_inf) / n_z

xs = np.linspace(start=x_inf, stop=x_sup, num=n_x + 1, endpoint=True, dtype=np.float64)
ys = np.linspace(start=y_inf, stop=y_sup, num=n_y + 1, endpoint=True, dtype=np.float64)
zs = np.linspace(start=z_inf, stop=z_sup, num=n_z + 1, endpoint=True, dtype=np.float64)

phi = np.ndarray(shape=(n_x + 1, n_y + 1, n_z + 1), dtype=np.float64)

# A simple elliptic interface.
a = 0.5
b = 0.5
c = np.sqrt(2)/4