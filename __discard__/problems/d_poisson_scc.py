import numpy as np
import config

# FOR DEBUG.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class poisson_scc(object):
    def __init__(self, interface):
        self.u_exact = np.ndarray(shape=(config.n_x + 1, config.n_y + 1, config.n_z + 1), dtype=np.float64)
        self.f = np.ndarray(shape=(config.n_x + 1, config.n_y + 1, config.n_z + 1), dtype=np.float64)

        self.interface = interface
    
        for i in range(config.n_x + 1):
            for j in range(config.n_y + 1):
                for k in range(config.n_z + 1):
                    self.u_exact[i, j, k] = self.__u_exact(i, j, k)
                    self.f[i, j, k] = self.__f(i, j, k)

    # Poisson equation.
    def __u_exact(self, i, j, k):
        if (self.interface.phi[i, j, k] >= 0):
            return np.cos(config.xs[i]) * np.sin(config.ys[j]) * np.sin(config.zs[k])
        return 0.0

    def __f(self, i, j, k):
        if (self.interface.phi[i, j, k] >= 0):
            return -3.0 * np.cos(config.xs[i]) * np.sin(config.ys[j]) * np.sin(config.zs[k])
        return 0.0

    # Jump conditions (neg->pos).
    # Assume the point locates on the interface (drop all the checking).
    def __jump_u(x, y, z):
        return np.cos(x) * np.sin(y) * np.sin(z)

    def __jump_u_x(x, y, z):
        return -np.sin(x) * np.sin(y) * np.sin(z)
    def __jump_u_y(x, y, z):
        return np.cos(x) * np.cos(y) * np.sin(z)
    def __jump_u_z(x, y, z):
        return np.cos(x) * np.sin(y) * np.cos(z)
    def __jump_u_n(self, x, y, z):
        abs_grad = np.sqrt(x**2 / config.a**4 + y**2 / config.b**4 + z**2 / config.c**4)
        abs_x = x/config.a**2 / abs_grad
        abs_y = y/config.b**2 / abs_grad
        abs_z = z/config.c**2 / abs_grad
        
        return self.jump_u_x(x, y, z)*abs_x + self.jump_u_y(x, y, z)*abs_y + self.jump_u_z(x, y, z)*abs_z

    def __jump_u_nn(self, x, y, z):
        abs_grad = np.sqrt(x**2 / config.a**4 + y**2 / config.b**4 + z**2 / config.c**4)
        abs_x = x/config.a**2 / abs_grad
        abs_y = y/config.b**2 / abs_grad
        abs_z = z/config.c**2 / abs_grad

        u_xx = -np.cos(x) * np.sin(y) * np.sin(z)
        u_yy = -np.cos(x) * np.sin(y) * np.sin(z)
        u_zz = -np.cos(x) * np.sin(y) * np.sin(z)
        u_xy = -np.sin(x) * np.cos(y) * np.sin(z)
        u_xz = -np.sin(x) * np.sin(y) * np.cos(z)
        u_yz = np.cos(x) * np.cos(y) * np.cos(z)
        return u_xx*abs_x*abs_x + u_yy*abs_y*abs_y + u_zz*abs_z*abs_z \
        + u_xy*abs_x*abs_y + u_xz*abs_x*abs_z + u_yz*abs_y*abs_z

    def __jump_f(x, y, z):
        return -3.0 * np.cos(x) * np.sin(y) * np.sin(z)


class interface_ellipsoid(object):
    def __init__(self):
        self.phi = np.zeros((config.n_x + 1, config.n_y + 1, config.n_z + 1), dtype=np.float64)
        self.irr = np.zeros((config.n_x + 1, config.n_y + 1, config.n_z + 1), dtype=np.int)
        
        for i in range(config.n_x + 1):
            for j in range(config.n_y + 1):
                for k in range(config.n_z + 1):
                    self.phi[i, j, k] = self.__phi(config.xs[i], config.ys[j], config.zs[k])
                    
        # Irregular points are sought after phi is calculated.
        self.n_irr = 0
        for i in range(config.n_x + 1):
            for j in range(config.n_y + 1):
                for k in range(config.n_z + 1):
                    if (i == 0 or i == config.n_x or
                        j == 0 or j == config.n_y or
                        k == 0 or k == config.n_z):
                        continue
                    
                    if (self.phi[i, j, k] == 0.0):
                        self.irr[i, j, k] = 1
                        continue

                    dirs = [0, 0, 1]
                    for l in range(3):
                        dirs =  np.roll(dirs, 1)
                        if ((self.phi[i, j, k] <= 0.0 and self.phi[i + dirs[0], j + dirs[1], k + dirs[2]] > 0.0) or
                            (self.phi[i, j, k] > 0.0 and self.phi[i + dirs[0], j + dirs[1], k + dirs[2]] <= 0.0)):
                            self.irr[i, j, k] = 1
                        if ((self.phi[i, j, k] <= 0.0 and self.phi[i - dirs[0], j - dirs[1], k - dirs[2]] > 0.0) or
                            (self.phi[i, j, k] > 0.0 and self.phi[i - dirs[0], j - dirs[1], k - dirs[2]] <= 0.0)):
                            self.irr[i, j, k] = 1

                    if (self.irr[i, j, k] == 1):
                        self.n_irr = self.n_irr + 1
                        self.irr[i, j, k] = self.n_irr

        """ DEBUG
        fig = plt.figure()            
        ax = fig.gca(projection='3d')
        z = config.n_z//2 + 5
        z1 = self.phi[:,:,z]
        z2 = self.irr[:,:,z]
        print(config.zs[z])

        xs, ys = np.meshgrid(config.xs, config.ys)
        ax.plot_wireframe(xs, ys, z1, linewidth=1, color="red")
        ax.scatter(xs, ys, z2, linewidths=1)
        plt.show()
        """


    # Level set function (INACCURATE, iterative method for SDF can be used instead).
    @staticmethod
    def __phi(x, y, z):
        return x**2 / config.a**2 + y**2 / config.b**2 + z**2 / config.c**2 - 1.0


#interface_A = interface_ellipsoid()
#A = poisson_scc(interface_A)