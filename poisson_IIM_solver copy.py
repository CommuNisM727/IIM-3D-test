from operator import floordiv
import numpy as np

from poisson_scc import *
from utils.utils_basic import *

class poisson_IIM_solver(object):
    def __init__(self, pde):
        self.pde = pde
        self.__irregular_projection()

    def __irregular_projection(self):
        for i in range(1, self.pde.mesh.n_x):
            for j in range(1, self.pde.mesh.n_y):
                for k in range(1, self.pde.mesh.n_z):
                    if (self.pde.interface.irr[i, j, k] > 0):
                        index = self.pde.interface.irr[i, j, k]
                        print("index:", index)
                        print("orig:", i, j, k)
                        # Initialize a projection point P_0=(x_0, y_0, z_0)
                        [alpha, x_0, y_0, z_0] = self.__irregular_projection_initial(i, j, k)
                        
                        print("proj:", x_0, y_0, z_0)
                        #[dis, x_k, y_k, z_k] = 
                        self.__irregular_projection_iterate(i, j, k, x_0, y_0, z_0, alpha)

    def __irregular_projection_initial(self, i, j, k):
        # Initial orthogonal projection from [i, j, k] to interface. (root of 2nd taylor expansion of phi)
        phi_x = self.pde.interface.phi_x_(i, j, k, self.pde.mesh.h_x)
        phi_y = self.pde.interface.phi_y_(i, j, k, self.pde.mesh.h_y)
        phi_z = self.pde.interface.phi_z_(i, j, k, self.pde.mesh.h_z)
        norm_phi = np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)
        # Normalize the gradient.
        phi_x = phi_x / norm_phi
        phi_y = phi_y / norm_phi
        phi_z = phi_z / norm_phi

        phi_xx = self.pde.interface.phi_xx_(i, j, k, h_x=self.pde.mesh.h_x)
        phi_xy = self.pde.interface.phi_xy_(i, j, k, h_x=self.pde.mesh.h_x, h_y=self.pde.mesh.h_y)
        phi_xz = self.pde.interface.phi_xz_(i, j, k, h_x=self.pde.mesh.h_x, h_z=self.pde.mesh.h_z)
        
        phi_yx = self.pde.interface.phi_yx_(i, j, k, h_x=self.pde.mesh.h_x, h_y=self.pde.mesh.h_y)
        phi_yy = self.pde.interface.phi_yy_(i, j, k, h_y=self.pde.mesh.h_y)
        phi_yz = self.pde.interface.phi_yz_(i, j, k, h_y=self.pde.mesh.h_y, h_z=self.pde.mesh.h_z)
        
        phi_zx = self.pde.interface.phi_zx_(i, j, k, h_x=self.pde.mesh.h_x, h_z=self.pde.mesh.h_z)
        phi_zy = self.pde.interface.phi_zy_(i, j, k, h_y=self.pde.mesh.h_y, h_z=self.pde.mesh.h_z)
        phi_zz = self.pde.interface.phi_zz_(i, j, k, h_z=self.pde.mesh.h_z)
        
        # Quadratic form (IIM overview [Li], CLAIMED to have 3rd acc).
        # phi(x) + |\nabla phi| * \alpha +  \alpha^2/2 * (p^t Hess p) = 0
        a = phi_x * (phi_xx*phi_x + phi_xy*phi_y + phi_xz*phi_z)
        +   phi_y * (phi_yx*phi_x + phi_yy*phi_y + phi_yz*phi_z)
        +   phi_z * (phi_zx*phi_x + phi_zy*phi_y + phi_zz*phi_z)
        b = norm_phi
        c = self.pde.interface.phi[i, j, k]

        print("point: ", self.pde.mesh.xs[i], self.pde.mesh.ys[j], self.pde.mesh.zs[k])
        print("grad: ", phi_x, phi_y, phi_z)

        [r1, r2] = root_p2(0.5 * a, b, c)
        if (np.abs(r1) <= np.abs(r2)):
            return [r1, self.pde.mesh.xs[i] + r1 * phi_x,
                        self.pde.mesh.ys[j] + r1 * phi_y,
                        self.pde.mesh.zs[k] + r1 * phi_z]
        else:
            return [r2, self.pde.mesh.xs[i] + r2 * phi_x,
                        self.pde.mesh.ys[j] + r2 * phi_y,
                        self.pde.mesh.zs[k] + r2 * phi_z]

    def __irregular_projection_iterate(self, i_0, j_0, k_0, x_0, y_0, z_0, alpha, iter_max=50, eps=1e-10):
        # Project from (x_fix, y_fix, z_fix).
        x_fix = self.pde.mesh.xs[i_0]
        y_fix = self.pde.mesh.xs[j_0]
        z_fix = self.pde.mesh.xs[k_0]
        
        # Iteration variable (x, y, z, lambda_).
        # Initial guess of projection point (x_k, y_k, z_k) obtained from quadratic form.
        x_k = x_0
        y_k = y_0
        z_k = z_0
        # Initial guess of distance |phi|_0 = alpha / |grad phi(x_0, y_0, z_0)|. 
        norm_grad_phi_k = alpha / self.__norm_grad_phi(x_k, y_k, z_k)
        print("norm_grad_phi_0: ", norm_grad_phi_k)

        for i in range(iter_max):
            print("ITR", i)
            delta = self.__iteration_newton_increasement(x_fix, y_fix, z_fix, x_k, y_k, z_k, norm_grad_phi_k)
            x_k = x_k - delta[0]
            y_k = y_k - delta[1]
            z_k = z_k - delta[2]
            norm_grad_phi_k = norm_grad_phi_k - delta[3]

            print("DELTA: ", delta)
            print("proj at: ", x_k, y_k, z_k)
            input()

            if (np.linalg.norm(delta) < eps):
                break

        # Interpolate 

    def __iteration_newton_increasement(self, x_fix, y_fix, z_fix, x_k, y_k, z_k, lambda_):
        # Interpolate all derivatives of point (x_k, y_k, z_k).
        i = int((x_k - self.pde.mesh.x_inf) / self.pde.mesh.h_x)
        j = int((y_k - self.pde.mesh.y_inf) / self.pde.mesh.h_y)
        k = int((z_k - self.pde.mesh.z_inf) / self.pde.mesh.h_z)
        print("proj in: ", i, j, k)

        ratio_x = (x_k - i * self.pde.mesh.h_x - self.pde.mesh.x_inf) / self.pde.mesh.h_x
        ratio_y = (y_k - j * self.pde.mesh.h_y - self.pde.mesh.y_inf) / self.pde.mesh.h_y
        ratio_z = (z_k - k * self.pde.mesh.h_z - self.pde.mesh.z_inf) / self.pde.mesh.h_z
        
        """
        n = 2
        offset = -int((n - 1) / 2)
        
        phi = Pn_interp_3d(x_k, y_k, z_k, 
                            self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs, 
                            i + offset, j + offset, k + offset, 
                            self.pde.interface.phi, n)
        """
        #"""
        phi_s = self.__phi_n(i, j, k, 2)
        phi = linear_interp_3d(vals=phi_s, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        #"""
        print("Rx, Ry, Rz: ", ratio_x, ratio_y, ratio_z)

        print("PHI: ", phi)
        

        # Generate all derivatives of phi in each directions for tri-linear interpolating.
        phi_xs = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_x_, self.pde.mesh.h_x)
        phi_ys = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_y_, self.pde.mesh.h_y)
        phi_zs = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_z_, self.pde.mesh.h_z)

        phi_x = linear_interp_3d(vals=phi_xs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_y = linear_interp_3d(vals=phi_ys, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_z = linear_interp_3d(vals=phi_zs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)


        
        print(phi_x, phi_y, phi_z)

        """
        #phi_x = linear_interp_3d(vals=phi_xs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        #phi_y = linear_interp_3d(vals=phi_ys, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        #phi_z = linear_interp_3d(vals=phi_zs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        n = 2
        phi_x = Pn_interp_3d(x_k, y_k, z_k, 
                            self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs, 
                            0, 0, 0, 
                            phi_xs, n)
        phi_y = Pn_interp_3d(x_k, y_k, z_k, 
                            self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs, 
                            0, 0, 0, 
                            phi_ys, n)
        phi_z = Pn_interp_3d(x_k, y_k, z_k, 
                            self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs, 
                            0, 0, 0, 
                            phi_zs, n)
        """

        phi_xxs = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_xx_, self.pde.mesh.h_x)
        phi_xys = self.__phi_derivs_2(i, j, k, self.pde.interface.phi_xy_, self.pde.mesh.h_x, self.pde.mesh.h_y)
        phi_xzs = self.__phi_derivs_2(i, j, k, self.pde.interface.phi_xz_, self.pde.mesh.h_x, self.pde.mesh.h_z)
        
        phi_yxs = self.__phi_derivs_2(i, j, k, self.pde.interface.phi_yx_, self.pde.mesh.h_x, self.pde.mesh.h_y)
        phi_yys = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_yy_, self.pde.mesh.h_y)
        phi_yzs = self.__phi_derivs_2(i, j, k, self.pde.interface.phi_yz_, self.pde.mesh.h_y, self.pde.mesh.h_z)
        
        phi_zxs = self.__phi_derivs_2(i, j, k, self.pde.interface.phi_zx_, self.pde.mesh.h_x, self.pde.mesh.h_z)
        phi_zys = self.__phi_derivs_2(i, j, k, self.pde.interface.phi_zy_, self.pde.mesh.h_y, self.pde.mesh.h_z)
        phi_zzs = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_zz_, self.pde.mesh.h_z)
        
        phi_xx = linear_interp_3d(vals=phi_xxs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_xy = linear_interp_3d(vals=phi_xys, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_xz = linear_interp_3d(vals=phi_xzs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_yx = linear_interp_3d(vals=phi_yxs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_yy = linear_interp_3d(vals=phi_yys, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_yz = linear_interp_3d(vals=phi_yzs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_zx = linear_interp_3d(vals=phi_zxs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_zy = linear_interp_3d(vals=phi_zys, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_zz = linear_interp_3d(vals=phi_zzs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)


        f = np.ndarray(shape=(4,), dtype=np.float64)
        Df = np.ndarray(shape=(4, 4), dtype=np.float64)

        f[0] = x_fix - x_k + lambda_*phi_x
        f[1] = y_fix - y_k + lambda_*phi_y
        f[2] = z_fix - z_k + lambda_*phi_z
        f[3] = phi
        print("f:", f)

        Df[0, 0] = lambda_*phi_xx - 1.0
        Df[0, 1] = lambda_*phi_xy
        Df[0, 2] = lambda_*phi_xz
        Df[0, 3] = phi_x
        
        Df[1, 0] = lambda_*phi_yx
        Df[1, 1] = lambda_*phi_yy - 1.0
        Df[1, 2] = lambda_*phi_yz
        Df[1, 3] = phi_y

        Df[2, 0] = lambda_*phi_zx
        Df[2, 1] = lambda_*phi_zy
        Df[2, 2] = lambda_*phi_zz - 1.0
        Df[2, 3] = phi_z

        Df[3, 0] = phi_x
        Df[3, 1] = phi_y
        Df[3, 2] = phi_z
        Df[3, 3] = 0.0
        
        return np.linalg.solve(Df, f)


    def __norm_grad_phi(self, x, y, z):
        i = int((x - self.pde.mesh.x_inf) / self.pde.mesh.h_x)
        j = int((y - self.pde.mesh.y_inf) / self.pde.mesh.h_y)
        k = int((z - self.pde.mesh.z_inf) / self.pde.mesh.h_z)

        ratio_x = (x - i * self.pde.mesh.h_x - self.pde.mesh.x_inf) / self.pde.mesh.h_x
        ratio_y = (y - j * self.pde.mesh.h_y - self.pde.mesh.y_inf) / self.pde.mesh.h_y
        ratio_z = (z - k * self.pde.mesh.h_z - self.pde.mesh.z_inf) / self.pde.mesh.h_z
        
        # Generate derivatives of phi in each directions for tri-linear interpolating.
        phi_xs = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_x_, self.pde.mesh.h_x)
        phi_ys = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_y_, self.pde.mesh.h_y)
        phi_zs = self.__phi_derivs_1(i, j, k, self.pde.interface.phi_z_, self.pde.mesh.h_z)

        phi_x = linear_interp_3d(vals=phi_xs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_y = linear_interp_3d(vals=phi_ys, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)
        phi_z = linear_interp_3d(vals=phi_zs, ratio_x=ratio_x, ratio_y=ratio_y, ratio_z=ratio_z)

        print("NPXYZ:", phi_x, phi_y, phi_z)

        return np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)

    def __phi_n(self, i, j, k, n):
        # n > 2.
        phi_n = np.ndarray(shape=(n, n, n), dtype=np.float64)
        # quasi-symmetric (a bit heavier on right side).
        offset = int((n - 1) / 2)
        i = i - offset
        j = j - offset
        k = k - offset

        for i_ in range(n):
            for j_ in range(n):
                for k_ in range(n):
                    phi_n[i_, j_, k_] = self.pde.interface.phi[i + i_, j + j_, k + k_]
            
        return phi_n

    def __phi_derivs_1(self, i, j, k, func, h):
        # Return 8 interpolating data in cubic (i, j, k)->(i + 1, j + 1, k + 1).
        phi_derivs = np.ndarray(shape=(2, 2, 2), dtype=np.float64)
        phi_derivs[0, 0, 0] = func(i, j, k, h)
        phi_derivs[1, 0, 0] = func(i + 1, j, k, h)
        phi_derivs[0, 1, 0] = func(i, j + 1, k, h)
        phi_derivs[0, 0, 1] = func(i, j, k + 1, h)
        phi_derivs[1, 1, 0] = func(i + 1, j + 1, k, h)
        phi_derivs[1, 0, 1] = func(i + 1, j, k + 1, h)
        phi_derivs[0, 1, 1] = func(i, j + 1, k + 1, h)
        phi_derivs[1, 1, 1] = func(i + 1, j + 1, k + 1, h)
        return phi_derivs

    def __phi_derivs_2(self, i, j, k, func, h1, h2):
        # Return 8 interpolating data in cubic (i, j, k)->(i + 1, j + 1, k + 1).
        phi_derivs = np.ndarray(shape=(2, 2, 2), dtype=np.float64)
        phi_derivs[0, 0, 0] = func(i, j, k, h1, h2)
        phi_derivs[1, 0, 0] = func(i + 1, j, k, h1, h2)
        phi_derivs[0, 1, 0] = func(i, j + 1, k, h1, h2)
        phi_derivs[0, 0, 1] = func(i, j, k + 1, h1, h2)
        phi_derivs[1, 1, 0] = func(i + 1, j + 1, k, h1, h2)
        phi_derivs[1, 0, 1] = func(i + 1, j, k + 1, h1, h2)
        phi_derivs[0, 1, 1] = func(i, j + 1, k + 1, h1, h2)
        phi_derivs[1, 1, 1] = func(i + 1, j + 1, k + 1, h1, h2)
        return phi_derivs


mesh = mesh_uniform(multiplier=1)
inte = interface_ellipsoid(0.6, 0.5, 0.4, mesh)
a = poisson_scc(inte, mesh)
scc = poisson_IIM_solver(a)