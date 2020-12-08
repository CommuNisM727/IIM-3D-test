import numpy as np
import utils.helmholtz3D

from poisson_scc import *
from utils.utils_basic import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class poisson_IIM_solver(object):
    def __init__(self, pde):
        self.pde = pde
        
        self.irr_proj = np.ndarray(shape=(self.pde.interface.n_irr + 1, 3), dtype=np.float64)
        self.irr_dist = np.ndarray(shape=(self.pde.interface.n_irr + 1, ), dtype=np.float64)
        self.irr_Xi = np.ndarray(shape=(self.pde.interface.n_irr + 1, 3), dtype=np.float64)
        self.irr_Eta = np.ndarray(shape=(self.pde.interface.n_irr + 1, 3), dtype=np.float64)
        self.irr_Tau = np.ndarray(shape=(self.pde.interface.n_irr + 1, 3), dtype=np.float64)
        self.irr_Kappa = np.ndarray(shape=(self.pde.interface.n_irr + 1, ), dtype=np.float64)

        self.irr_jump_u = np.ndarray(shape=(self.pde.interface.n_irr + 1, ), dtype=np.float64)
        self.irr_jump_f = np.ndarray(shape=(self.pde.interface.n_irr + 1, ), dtype=np.float64)
        self.irr_jump_u_n = np.ndarray(shape=(self.pde.interface.n_irr + 1, ), dtype=np.float64)
        self.irr_jump_u_nn = np.ndarray(shape=(self.pde.interface.n_irr + 1, ), dtype=np.float64)

        self.irr_corr = np.zeros(shape=(self.pde.interface.n_irr + 1, ), dtype=np.float64)
        
        self.u = np.asfortranarray(np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_y + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F'))
        self.error = 0.0


        self.__irregular_projection()
        self.__solve()
        self.__error_estimate()

    def __irregular_projection(self):
        # Find the projections & basic curve information.
        for i in range(1, self.pde.mesh.n_x):
            for j in range(1, self.pde.mesh.n_y):
                for k in range(1, self.pde.mesh.n_z):
                    if (self.pde.interface.irr[i, j, k] > 0):
                        index = self.pde.interface.irr[i, j, k]
                        #print("index:", index)

                        # Initialize the projection P_0=(x_0, y_0, z_0)
                        [alpha, x_0, y_0, z_0] = self.__irregular_projection_initial(i, j, k)
                        # Apply several Newton iterations to obtain the final projection P_k=(x_k, y_k, z_k)
                        [alpha, x_p, y_p, z_p] = self.__irregular_projection_iterate(i, j, k, x_0, y_0, z_0, alpha)

                        self.__irregular_projection_info(index, x_p, y_p, z_p, alpha)

        # 1. Find the second order normal derivatives of projections on the interface.
        # 2. Calculate the correction terms.
        for i in range(1, self.pde.mesh.n_x):
            for j in range(1, self.pde.mesh.n_y):
                for k in range(1, self.pde.mesh.n_z):
                    if (self.pde.interface.irr[i, j, k] > 0):
                        index = self.pde.interface.irr[i, j, k]
                        #print("index:", index)
                        self.__irregular_projection_jump(index, i, j, k)
                        self.__irregular_projection_corr(index, i, j, k)

    def __irregular_projection_initial(self, i, j, k):
        # Initial orthogonal projection from [i, j, k] to interface. (root of 2nd taylor expansion of phi)
        phi_x = self.pde.interface.phi_x[i, j, k]
        phi_y = self.pde.interface.phi_y[i, j, k]   
        phi_z = self.pde.interface.phi_z[i, j, k]           
        norm_grad_phi = np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)
        # Normalize the gradient.
        phi_x = phi_x / norm_grad_phi
        phi_y = phi_y / norm_grad_phi
        phi_z = phi_z / norm_grad_phi

        phi_xx = self.pde.interface.phi_xx[i, j, k]
        phi_xy = self.pde.interface.phi_xy[i, j, k]
        phi_xz = self.pde.interface.phi_xz[i, j, k]
        
        phi_yx = self.pde.interface.phi_yx[i, j, k]
        phi_yy = self.pde.interface.phi_yy[i, j, k]
        phi_yz = self.pde.interface.phi_yz[i, j, k]
        
        phi_zx = self.pde.interface.phi_zx[i, j, k]
        phi_zy = self.pde.interface.phi_zy[i, j, k]
        phi_zz = self.pde.interface.phi_zz[i, j, k]
        
        # Quadratic form (IIM overview [Li], CLAIMED to have 3rd acc).
        # phi(x) + |\nabla phi| * \alpha +  \alpha^2/2 * (p^t Hess p) = 0
        a = phi_x * (phi_xx*phi_x + phi_xy*phi_y + phi_xz*phi_z)
        +   phi_y * (phi_yx*phi_x + phi_yy*phi_y + phi_yz*phi_z)
        +   phi_z * (phi_zx*phi_x + phi_zy*phi_y + phi_zz*phi_z)
        b = norm_grad_phi
        c = self.pde.interface.phi[i, j, k]

        #print("point: ", self.pde.mesh.xs[i], self.pde.mesh.ys[j], self.pde.mesh.zs[k])

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
        y_fix = self.pde.mesh.ys[j_0]
        z_fix = self.pde.mesh.zs[k_0]
        
        # Iteration variable (x, y, z, phi).
        # Initial guess of projection point (x_k, y_k, z_k) obtained from quadratic form.
        x_k = x_0
        y_k = y_0
        z_k = z_0
        # Initial guess of distance |phi|_0 = alpha / |grad phi(x_0, y_0, z_0)|. 
        norm_grad_phi_k = alpha / self.__norm_grad_phi(x_k, y_k, z_k)

        for i in range(iter_max):
            #print("ITR", i)
            delta = self.__newton_increasement(x_fix, y_fix, z_fix, x_k, y_k, z_k, norm_grad_phi_k)
            #print("DELTA: ", delta)
            
            x_k = x_k - delta[0]
            y_k = y_k - delta[1]
            z_k = z_k - delta[2]
            norm_grad_phi_k = norm_grad_phi_k - delta[3]

            if (np.linalg.norm(delta) < eps):
                break
            
        return [norm_grad_phi_k * self.__norm_grad_phi(x_k, y_k, z_k), x_k, y_k, z_k]

    def __newton_increasement(self, x_fix, y_fix, z_fix, x_k, y_k, z_k, lambda_):
        # Interpolate all derivatives of point (x_k, y_k, z_k).
        i = int((x_k - self.pde.mesh.x_inf) / self.pde.mesh.h_x)
        j = int((y_k - self.pde.mesh.y_inf) / self.pde.mesh.h_y)
        k = int((z_k - self.pde.mesh.z_inf) / self.pde.mesh.h_z)

        Pn_interp_3d_partial = lambda arr, n: Pn_interp_3d(x_k, y_k, z_k, 
                                                self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs,
                                                i - int((n - 1) / 2), j - int((n - 1) / 2), k - int((n - 1) / 2),
                                                arr, n)
        # Tricubic interpolating.
        phi = Pn_interp_3d_partial(self.pde.interface.phi, 4)

        # Trilinear interpolating.
        phi_x, phi_y, phi_z = self.__phi_derivs1_trilinear(x_k, y_k, z_k, i, j, k)
        phi_xx, phi_xy, phi_xz, phi_yx, phi_yy, phi_yz, phi_zx, phi_zy, phi_zz = self.__phi_derivs2_trilinear(x_k, y_k, z_k, i, j, k)

        f = np.ndarray(shape=(4, ), dtype=np.float64)
        Df = np.ndarray(shape=(4, 4), dtype=np.float64)

        f[0] = x_fix - x_k + lambda_*phi_x
        f[1] = y_fix - y_k + lambda_*phi_y
        f[2] = z_fix - z_k + lambda_*phi_z
        f[3] = phi

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

        phi_x, phi_y, phi_z = self.__phi_derivs1_trilinear(x, y, z, i, j, k)

        return np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)

    def __irregular_projection_info(self, index, x, y, z, alpha):
        self.irr_proj[index, 0] = x
        self.irr_proj[index, 1] = y
        self.irr_proj[index, 2] = z
        self.irr_dist[index] = -alpha

        i = int((x - self.pde.mesh.x_inf) / self.pde.mesh.h_x)
        j = int((y - self.pde.mesh.y_inf) / self.pde.mesh.h_y)
        k = int((z - self.pde.mesh.z_inf) / self.pde.mesh.h_z)

        phi_x, phi_y, phi_z = self.__phi_derivs1_trilinear(x, y, z, i, j, k)
        phi_xx, phi_xy, phi_xz, phi_yx, phi_yy, phi_yz, phi_zx, phi_zy, phi_zz = self.__phi_derivs2_trilinear(x, y, z, i, j, k)

        norm_grad_phi = np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)

        # Normalize the gradient.
        phi_x_nm = phi_x / norm_grad_phi
        phi_y_nm = phi_y / norm_grad_phi
        phi_z_nm = phi_z / norm_grad_phi

        # Local coordinates.
        self.irr_Xi[index, 0] = phi_x_nm
        self.irr_Xi[index, 1] = phi_y_nm
        self.irr_Xi[index, 2] = phi_z_nm

        if (np.abs(phi_y_nm) > np.abs(phi_z_nm)):
            norm = np.sqrt(phi_x_nm**2 + phi_y_nm**2)
            self.irr_Eta[index, 0] = phi_y_nm/norm
            self.irr_Eta[index, 1] = -phi_x_nm/norm
            self.irr_Eta[index, 2] = 0.0

            norm = np.sqrt((phi_x_nm*phi_z_nm)**2 + (phi_y_nm*phi_z_nm)**2 + (phi_x_nm**2 + phi_y_nm**2)**2)
            self.irr_Tau[index, 0] = phi_x_nm*phi_z_nm/norm
            self.irr_Tau[index, 1] = phi_y_nm*phi_z_nm/norm
            self.irr_Tau[index, 2] = -(phi_x_nm**2 + phi_y_nm**2)/norm
        else:
            norm = np.sqrt(phi_x_nm**2 + phi_z_nm**2)
            self.irr_Eta[index, 0] = phi_z_nm/norm
            self.irr_Eta[index, 1] = 0.0
            self.irr_Eta[index, 2] = -phi_x_nm/norm

            norm = np.sqrt((phi_x_nm*phi_y_nm)**2 + (phi_x_nm**2 + phi_z_nm**2)**2 + (phi_y_nm*phi_z_nm)**2)
            self.irr_Tau[index, 0] = -phi_x_nm*phi_y_nm/norm
            self.irr_Tau[index, 1] = (phi_x_nm**2 + phi_z_nm**2)/norm
            self.irr_Tau[index, 2] = -phi_y_nm*phi_z_nm/norm
            
        # Mean curvature. (https://math.mit.edu/classes/18.086/2007/levelsetnotes.pdf)
        self.irr_Kappa[index] = phi_xx * (phi_y*phi_y + phi_z*phi_z)
        +                       phi_yy * (phi_x*phi_x + phi_z*phi_z) 
        +                       phi_zz * (phi_x*phi_x + phi_y*phi_y)
        -                       phi_x * (phi_xy*phi_y + phi_xz*phi_z)
        -                       phi_y * (phi_yx*phi_x + phi_yz*phi_z)
        -                       phi_z * (phi_zx*phi_x + phi_zy*phi_y)
        self.irr_Kappa[index] = self.irr_Kappa[index] / (phi_x**2+phi_y**2+phi_z**2) / np.sqrt(phi_x**2 + phi_y**2 + phi_z**2)
    
        self.irr_jump_u[index] = self.pde.jump_u(x, y, z)
        self.irr_jump_f[index] = self.pde.jump_f(x, y, z)
        self.irr_jump_u_n[index] = self.pde.jump_u_n(x, y, z)

    def __irregular_projection_jump(self, index, i, j, k, norm_l1=3, norm_l2=2.4, n_points=12):
        x = self.irr_proj[index, 0]
        y = self.irr_proj[index, 1]
        z = self.irr_proj[index, 2]

        neighbours = np.ndarray(shape=(n_points, ), dtype=np.int)
        distances = np.ndarray(shape=(n_points, ), dtype=np.float64)
        
        n = 0
        for offset_i in range(-norm_l1, norm_l1 + 1):
            for offset_j in range(-norm_l1, norm_l1 + 1):
                for offset_k in range(-norm_l1, norm_l1 + 1):
                    # Neighbour point index.
                    i_ = i + offset_i
                    j_ = j + offset_j
                    k_ = k + offset_k

                    if ((n < n_points) and (self.pde.interface.irr[i_, j_, k_] > 0)):
                        index_ = self.pde.interface.irr[i_, j_, k_]
                        x_ = self.irr_proj[index_, 0]
                        y_ = self.irr_proj[index_, 1]
                        z_ = self.irr_proj[index_, 2]

                        dist = np.sqrt((x-x_)**2 + (y-y_)**2 + (z-z_)**2)
                        if (dist <= norm_l2*self.pde.mesh.h_x):
                            neighbours[n] = index_
                            distances[n] = dist
                            n = n + 1
        
        order = np.argsort(distances)
        neighbours = neighbours[order]

        neighbour_jump_u = np.ndarray(shape=(n_points, ), dtype=np.float64)
        neighbour_Eta_proj = np.ndarray(shape=(n_points, ), dtype=np.float64)
        neighbour_Tau_proj = np.ndarray(shape=(n_points, ), dtype=np.float64)
        
        # n_features * n_points.
        n_features = 15
        neighbour_dict = np.ndarray(shape=(n_features, n_points), dtype=np.float64)

        for n_ in range(n):
            index_ = neighbours[n_]
            dx = self.irr_proj[index_, 0] - x
            dy = self.irr_proj[index_, 1] - y
            dz = self.irr_proj[index_, 2] - z
            
            neighbour_jump_u[n_] = self.irr_jump_u[index_]
            neighbour_Eta_proj[n_] = self.irr_Eta[index, 0] * dx + self.irr_Eta[index, 1] * dy + self.irr_Eta[index, 2] * dz
            neighbour_Tau_proj[n_] = self.irr_Tau[index, 0] * dx + self.irr_Tau[index, 1] * dy + self.irr_Tau[index, 2] * dz
        
            Eta = neighbour_Eta_proj[n_]
            Tau = neighbour_Tau_proj[n_]

            # o0
            neighbour_dict[0, n_] = 1.0
            # o1
            neighbour_dict[1, n_] = Eta
            neighbour_dict[2, n_] = Tau
            # o2
            neighbour_dict[3, n_] = 0.5*Eta*Eta #
            neighbour_dict[4, n_] = Eta*Tau
            neighbour_dict[5, n_] = 0.5*Tau*Tau #
            # o3
            neighbour_dict[6, n_] = Eta*Eta*Eta
            neighbour_dict[7, n_] = Eta*Eta*Tau
            neighbour_dict[8, n_] = Eta*Tau*Tau
            neighbour_dict[9, n_] = Tau*Tau*Tau
            # o4
            neighbour_dict[10, n_] = Eta*Eta*Eta*Eta
            neighbour_dict[11, n_] = Eta*Eta*Eta*Tau
            neighbour_dict[12, n_] = Eta*Eta*Tau*Tau
            neighbour_dict[13, n_] = Eta*Tau*Tau*Tau
            neighbour_dict[14, n_] = Tau*Tau*Tau*Tau
            
        
        derivs = np.dot(np.linalg.pinv(np.transpose(neighbour_dict)), neighbour_jump_u)
        self.irr_jump_u_nn[index] = self.irr_jump_u_n[index] 
        - self.irr_Kappa * self.irr_jump_u_n 
        - derivs[3] - derivs[5]

    def __irregular_projection_corr(self, index, i, j, k):
        jump_u = self.irr_jump_u[index]
        jump_n_n = self.irr_jump_u_n[index]
        jump_u_nn = self.irr_jump_u_nn[index]
        d = self.irr_dist[index]
        
        corr = jump_u + d*jump_n_n + 0.5*d*d*jump_u_nn
        
        # x-.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i - 1, j, k] > 0):
            index_ = self.pde.interface.irr[i - 1, j, k]
            self.irr_corr[index_] = self.irr_corr[index_] + corr / self.pde.mesh.h_x**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i - 1, j, k] <= 0):
            index_ = self.pde.interface.irr[i - 1, j, k]
            self.irr_corr[index_] = self.irr_corr[index_] - corr / self.pde.mesh.h_x**2
        # x+.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i + 1, j, k] > 0):
            index_ = self.pde.interface.irr[i + 1, j, k]
            self.irr_corr[index_] = self.irr_corr[index_] + corr / self.pde.mesh.h_x**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i + 1, j, k] <= 0):
            index_ = self.pde.interface.irr[i + 1, j, k]
            self.irr_corr[index_] = self.irr_corr[index_] - corr / self.pde.mesh.h_x**2
        
        # y-.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j - 1, k] > 0):
            index_ = self.pde.interface.irr[i, j - 1, k]
            self.irr_corr[index_] = self.irr_corr[index_] + corr / self.pde.mesh.h_y**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j - 1, k] <= 0):
            index_ = self.pde.interface.irr[i, j - 1, k]
            self.irr_corr[index_] = self.irr_corr[index_] - corr / self.pde.mesh.h_y**2
        # y+.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j + 1, k] > 0):
            index_ = self.pde.interface.irr[i, j + 1, k]
            self.irr_corr[index_] = self.irr_corr[index_] + corr / self.pde.mesh.h_y**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j + 1, k] <= 0):
            index_ = self.pde.interface.irr[i, j + 1, k]
            self.irr_corr[index_] = self.irr_corr[index_] - corr / self.pde.mesh.h_y**2
        
        # z-.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j, k - 1] > 0):
            index_ = self.pde.interface.irr[i, j, k - 1]
            self.irr_corr[index_] = self.irr_corr[index_] + corr / self.pde.mesh.h_z**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j, k - 1] <= 0):
            index_ = self.pde.interface.irr[i, j, k - 1]
            self.irr_corr[index_] = self.irr_corr[index_] - corr / self.pde.mesh.h_z**2
        # z+.
        if (self.pde.interface.phi[i, j, k] <= 0 and self.pde.interface.phi[i, j, k + 1] > 0):
            index_ = self.pde.interface.irr[i, j, k + 1]
            self.irr_corr[index_] = self.irr_corr[index_] + corr / self.pde.mesh.h_z**2
        if (self.pde.interface.phi[i, j, k] > 0 and self.pde.interface.phi[i, j, k + 1] <= 0):
            index_ = self.pde.interface.irr[i, j, k + 1]
            self.irr_corr[index_] = self.irr_corr[index_] - corr / self.pde.mesh.h_z**2
        
    def __solve(self):
        for i in range(self.pde.mesh.n_x + 1):
            for j in range(self.pde.mesh.n_y + 1):
                for k in range(self.pde.mesh.n_z + 1):
                    # RHS initialization & modification.
                    self.u[i, j, k] = self.pde.f_exact[i, j, k]
                    
                    if (self.pde.interface.irr[i, j, k] > 0):
                        self.u[i, j, k] = self.u[i, j, k] - self.irr_corr[self.pde.interface.irr[i, j, k]]
                    
                    # Boundary conditions.
                    if (i == 0 or i == self.pde.mesh.n_x or
                        j == 0 or j == self.pde.mesh.n_y or
                        k == 0 or k == self.pde.mesh.n_z):
                            self.u[i, j, k] = self.pde.u_exact[i, j, k]


        # Dummy arrays.
        BDXS = np.zeros(shape=(self.pde.mesh.n_y + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDXF = np.zeros(shape=(self.pde.mesh.n_y + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDYS = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDYF = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_z + 1), dtype=np.float64, order='F')
        BDZS = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_y + 1), dtype=np.float64, order='F')
        BDZF = np.zeros(shape=(self.pde.mesh.n_x + 1, self.pde.mesh.n_y + 1), dtype=np.float64, order='F')
        PERTRB = np.array(0, dtype=np.float64)
        IERROR = np.array(-1, dtype=np.int32)
        W = np.zeros(shape=(
            30 + self.pde.mesh.n_x + self.pde.mesh.n_y + 5*self.pde.mesh.n_z 
            + np.max([self.pde.mesh.n_x, self.pde.mesh.n_y, self.pde.mesh.n_z]) 
            + 7*((self.pde.mesh.n_x + 1)//2 + (self.pde.mesh.n_y + 1)//2)), dtype=np.float64)
    
        print("FORTRAN OUTPUT:")
        #"""
        utils.helmholtz3D.hw3crtt(
            xs=np.array(self.pde.mesh.x_inf, dtype=np.float64), xf=np.array(self.pde.mesh.x_sup, dtype=np.float64), l=np.array(self.pde.mesh.n_x, dtype=np.int32), lbdcnd=np.array(1, dtype=np.int32), bdxs=BDXS, bdxf=BDXF,
            ys=np.array(self.pde.mesh.y_inf, dtype=np.float64), yf=np.array(self.pde.mesh.y_sup, dtype=np.float64), m=np.array(self.pde.mesh.n_y, dtype=np.int32), mbdcnd=np.array(1, dtype=np.int32), bdys=BDYS, bdyf=BDYF,
            zs=np.array(self.pde.mesh.z_inf, dtype=np.float64), zf=np.array(self.pde.mesh.z_sup, dtype=np.float64), n=np.array(self.pde.mesh.n_z, dtype=np.int32), nbdcnd=np.array(1, dtype=np.int32), bdzs=BDZS, bdzf=BDZF,
            elmbda=np.array(0, dtype=np.float64), f=self.u, pertrb=PERTRB, ierror=IERROR, w=W)
        #"""
        print("P, E:", PERTRB, IERROR)


        N = self.pde.mesh.n_x//2
        res = self.u[N, :, :] - self.pde.u_exact[N, :, :]
        Y, Z = np.meshgrid(self.pde.mesh.ys, self.pde.mesh.zs)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        #surf = ax.plot_surface(Y, Z, self.pde.u_exact[N, :, :], cmap=cm.coolwarm,
        #               linewidth=0, antialiased=False)

        surf = ax.plot_surface(Y, Z, res, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    def __error_estimate(self, dis_multiplier=1.5):
        for i in range(self.pde.mesh.n_x + 1):
            for j in range(self.pde.mesh.n_y + 1):
                for k in range(self.pde.mesh.n_z + 1):
                    if (np.abs(self.pde.interface.phi[i, j, k]) <= dis_multiplier*self.pde.mesh.h_x):
                        self.error = np.max([self.error, np.abs(self.u[i, j, k] - self.pde.u_exact[i, j, k])])

        print(self.error)

    # Estimate the derivatives of off-grid points.
    def __phi_derivs1_trilinear(self, x, y, z, i, j, k):
        P2_interp_3d = lambda arr: Pn_interp_3d(x, y, z, 
                                                self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs,
                                                i + 0, j + 0, k + 0,
                                                arr, 2)

        # Trilinear interpolating.
        phi_x = P2_interp_3d(self.pde.interface.phi_x)
        phi_y = P2_interp_3d(self.pde.interface.phi_y)
        phi_z = P2_interp_3d(self.pde.interface.phi_z)

        return [phi_x, phi_y, phi_z]

    def __phi_derivs2_trilinear(self, x, y, z, i, j, k):
        P2_interp_3d = lambda arr: Pn_interp_3d(x, y, z, 
                                                self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs,
                                                i + 0, j + 0, k + 0,
                                                arr, 2)

        phi_xx = P2_interp_3d(self.pde.interface.phi_xx)
        phi_xy = P2_interp_3d(self.pde.interface.phi_xy)
        phi_xz = P2_interp_3d(self.pde.interface.phi_xz)
        
        phi_yx = P2_interp_3d(self.pde.interface.phi_yx)
        phi_yy = P2_interp_3d(self.pde.interface.phi_yy)
        phi_yz = P2_interp_3d(self.pde.interface.phi_yz)
        
        phi_zx = P2_interp_3d(self.pde.interface.phi_zx)
        phi_zy = P2_interp_3d(self.pde.interface.phi_zy)
        phi_zz = P2_interp_3d(self.pde.interface.phi_zz)
        
        return [phi_xx, phi_xy, phi_xz, phi_yx, phi_yy, phi_yz, phi_zx, phi_zy, phi_zz]

mesh = mesh_uniform(multiplier=2)
inte = interface_ellipsoid(0.6, 0.5, 0.4, mesh)
a = poisson_scc(inte, mesh)
scc = poisson_IIM_solver(a)