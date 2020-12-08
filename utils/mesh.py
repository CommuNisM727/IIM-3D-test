import numpy as np

class mesh_uniform(object):
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