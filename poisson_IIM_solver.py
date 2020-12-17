### ALL REQUIRED PYTHON MODULES.
import numpy as np
import utils.helmholtz3D

from poisson_scc import *
from utils.utils_basic import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
###-----------------------------------------------------------------------------------
### FILE NAME:      poisson_IIM_solver.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    A 3D IIM solver.
### NOTED:          Might be split into multiple modules.
###-----------------------------------------------------------------------------------


class poisson_IIM_solver(object):
    """ A simple 3D IIM poisson solver.
    Attributes:
        pde         (pde object):       A poisson or helmholtz equation.
        irr_proj    (1D*3-array):       An array of projections of irregular points.
        irr_dist    (1D-array):         An array of distances to the interface of irregular points.
        
        irr_Xi      (1D*3-array):       An array of surface normal direction vector \Xi.
        irr_Eta     (1D*3-array):       An array of surface tangential vector \Eta.
        irr_Tau     (1D*3-array):       An array of surface tangential vector \Tau.
        irr_Kappa   (1D-array):         An array of surface mean curveture.
        irr_jump_u      (1D-array):         An array of [u].
        irr_jump_f      (1D-array):         An array of [f].
        irr_jump_u_n    (1D-array):         An array of [u_n].
        irr_jump_u_nn   (1D-array):         An array of [u_{nn}].
        irr_corr    (1D-array):         Correction terms on irregular points.
        u           (3D-array):         Numerical solution to the equation.
        error       (double):           Numerical error to the ground-truth. 
        
    """

    def __init__(self, pde):
        """ Initialization of class 'poisson IIM solver'.
            Initialization, solving, and error estimation are all done here.
        Args:
            pde         (pde object):       The PDE to be solved.
        Returns:
            None

        """

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
        """ The projections onto the interface of each irregular points are found here.
        Computation:
            1. Projections 'irr_proj' and distances 'irr_dist' are obtained in 2 steps:
            (1).  Directly draw a projection from solving quadratic equation as the initial point.
            (2).  Apply Newton iteration till the gradient on level-set function is normal to interface.
                irr_proj
                irr_dist
            
            2. Normal \Xi, tangential directions '\Eta, \Tau', curvature '\Kappa' at projecting points.
                irr_Xi      
                irr_Eta     
                irr_Tau     
                irr_Kappa
            3. All jumping conditions at projecting points.
                irr_jump_u  
                irr_jump_f  
                irr_jump_u_n
                irr_jump_u_nn
                irr_corr
        Args & Returns:
            None

        """

        # 1. Find the projections & basic curve information.
        for i in range(1, self.pde.mesh.n_x):
            for j in range(1, self.pde.mesh.n_y):
                for k in range(1, self.pde.mesh.n_z):
                    if (self.pde.interface.irr[i, j, k] > 0):
                        index = self.pde.interface.irr[i, j, k]
                        #print("index:", index)

                        # (1). Initialize the projection.
                        [alpha, x_0, y_0, z_0] = self.__irregular_projection_initial(i, j, k)
                        # (2). Apply several Newton iterations to obtain the final projection.
                        [alpha, x_p, y_p, z_p] = self.__irregular_projection_iterate(i, j, k, x_0, y_0, z_0, alpha)

                        # 2. Basic curve derivatives information.
                        self.__irregular_projection_info(index, x_p, y_p, z_p, alpha)

        # 2. Find the second order normal derivatives of projections on the interface.
        # 3. Calculate the correction terms.
        for i in range(1, self.pde.mesh.n_x):
            for j in range(1, self.pde.mesh.n_y):
                for k in range(1, self.pde.mesh.n_z):
                    if (self.pde.interface.irr[i, j, k] > 0):
                        index = self.pde.interface.irr[i, j, k]
                        #print("index:", index)
                        self.__irregular_projection_jump(index, i, j, k)
                        self.__irregular_projection_corr(index, i, j, k)

    def __irregular_projection_initial(self, i, j, k):
        """ A module for computing the projection. (AN OVERVIEW OF THE IMMERSED INTERFACE METHOD ... [Li.])
            For accurate SDF as level-set function \phi, we have x* = x - (\nabla \phi) \phi. (Eikonal equation)
            For arbitrary level-set function, we apply a second order Tylor-EXP approximation, which is solving:
                \phi + |\nabla \phi|\alpha + 1/2 p^T \nabla^2 \phi p \alpha^2 = 0,
            where x* = x + \alpha p, p = (\nabla\phi) / |\nabla\phi|.

        Args:
            i, j, k     (integer):      Index of irregular point to be projected.

        Returns:
            alpha:      (real):         Distance to the projecting point.
            x, y, z     (real):         Coords of the projecting point.

        """

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
        
        # Quadratic form (IIM overview [Li], CLAIMED to have 3rd ACC).
        # \phi + |\nabla \phi| * \alpha +  \alpha^2/2 * (p^t Hess p) = 0
        a = phi_x * (phi_xx*phi_x + phi_xy*phi_y + phi_xz*phi_z)    \
        +   phi_y * (phi_yx*phi_x + phi_yy*phi_y + phi_yz*phi_z)    \
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
        """ A module for computing the final projection.
            Newton fixed point iteration (x_{k+1} = x_k - Df^{-1} f):
                function(f):   (x_fix - x + \lambda*phi_x, y_fix - y + \lambda*phi_y, z_fix - z + \lambda*phi_z, phi).
                variable(x):   (x, y, z, \lambda).
            where \lambda is proportional to the projecting distance.

        Args:
            i_0, j_0, k_0   (integer):      Index of irregular point to be projected.
            x_0, y_0, z_0   (real):         Initial projecting point guess.

        Returns:
            alpha:          (real):         Distance to the projecting point.
            x, y, z         (real):         Coords of the projecting point.

        """

        # Project from (x_fix, y_fix, z_fix).
        x_fix = self.pde.mesh.xs[i_0]
        y_fix = self.pde.mesh.ys[j_0]
        z_fix = self.pde.mesh.zs[k_0]
        
        # Initial guess of projection point (x_k, y_k, z_k) obtained from solving quadratic equation.
        # Initial guess of distance \lambda = \alpha / |\nabla \phi(x_0, y_0, z_0)|. 
        x_k = x_0
        y_k = y_0
        z_k = z_0
        norm_grad_phi_k = alpha / self.__norm_grad_phi(x_0, y_0, z_0)

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
        """ A module for computing the increasement Df^{-1} f in Newton iteration.

        Args:
            x_fix, y_fix, z_fix     (real):     The coords of point to be projected.
            x_k, y_k, z_k           (real):     The coords of temporary projecting point.
            lambda_                 (real):     Proportional to the temporary projecting distance.

        Returns:
            delta                   (real*4):   The Newton increasement.

        """

        # Interpolate all derivatives of point (x_k, y_k, z_k).
        i = int((x_k - self.pde.mesh.x_inf) / self.pde.mesh.h_x)
        j = int((y_k - self.pde.mesh.y_inf) / self.pde.mesh.h_y)
        k = int((z_k - self.pde.mesh.z_inf) / self.pde.mesh.h_z)

        Pn_interp_3d_partial = lambda arr, n: Pn_interp_3d(x_k, y_k, z_k, 
                                                self.pde.mesh.xs, self.pde.mesh.ys, self.pde.mesh.zs,
                                                i - int((n - 1) / 2), j - int((n - 1) / 2), k - int((n - 1) / 2),
                                                arr, n)
        # Tricubic interpolating (4^3 points).
        phi = Pn_interp_3d_partial(self.pde.interface.phi, 4)

        # Trilinear interpolating (2^3 points).
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
        """ A module for computing the norm of gradient |\nabla \phi| at (x, y, z).

        Args:
            x, y, z     (real):     The coords of point.

        Returns:
            |\nabla \phi| at (x, y, z).

        """

        i = int((x - self.pde.mesh.x_inf) / self.pde.mesh.h_x)
        j = int((y - self.pde.mesh.y_inf) / self.pde.mesh.h_y)
        k = int((z - self.pde.mesh.z_inf) / self.pde.mesh.h_z)

        phi_x, phi_y, phi_z = self.__phi_derivs1_trilinear(x, y, z, i, j, k)

        return np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)

    def __irregular_projection_info(self, index, x, y, z, alpha):
        """ A module for computing and saving the basic information of irregular point projection on the interface.

        Args:
            index       (integer>0):    The index of irregular point.
            x, y, z     (real):         The coords of irregular point projection.
            alpha       (real):         The distance to the projection.

        Returns:
            None

        Computation:
            irr_proj, irr_dist are already found in previous steps, assignment only here.
            Directions \Xi, \Eta, \Tau and curvature \Kappa are calculated here (AN OVERVIEW OF THE IMMERSED INTERFACE METHOD ... [Li.])
            [u], [f], [u_n] are assigned here.

        """

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
        self.irr_Kappa[index] = phi_xx * (phi_y*phi_y + phi_z*phi_z)    \
        +                       phi_yy * (phi_x*phi_x + phi_z*phi_z)    \
        +                       phi_zz * (phi_x*phi_x + phi_y*phi_y)    \
        -                       phi_x * (phi_xy*phi_y + phi_xz*phi_z)   \
        -                       phi_y * (phi_yx*phi_x + phi_yz*phi_z)   \
        -                       phi_z * (phi_zx*phi_x + phi_zy*phi_y)   
        self.irr_Kappa[index] = self.irr_Kappa[index] / (phi_x**2+phi_y**2+phi_z**2) / np.sqrt(phi_x**2 + phi_y**2 + phi_z**2)
    
        self.irr_jump_u[index] = self.pde.jump_u(x, y, z)
        self.irr_jump_f[index] = self.pde.jump_f(x, y, z)
        self.irr_jump_u_n[index] = self.pde.jump_u_n(x, y, z)
        #self.irr_jump_u_n[index] = self.pde.jump_u_x(x, y, z) * phi_x_nm + self.pde.jump_u_y(x, y, z) * phi_y_nm + self.pde.jump_u_z(x, y, z) * phi_z_nm

    def __irregular_projection_jump(self, index, i, j, k, norm_l1=3, norm_l2=2.4, n_points=16):
        """ A module for computing [u_{nn}] using least square.

        Args:
            i, j, k     (integer):      The index of coords of irregular point.
            norm_l1     (real):         Neighbour search |area|_1 < norm_l1 * h.
            norm_l2     (real):         Neighbour search |area|_2 < norm_l2 * h.
            n_points    (integer):      Number of neighbour points used.

        Returns:
            None

        Computation:
            [u_{nn}] are assigned here. ([u_{nn}] = [f_n] - \Kappa [u_n] - [u_{surface Laplacian}])

        """

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
        
        # n_features (fixed) * n_points.
        n_features = 15
        neighbour_dict = np.ndarray(shape=(n_features, n_points), dtype=np.float64)
        neighbour_dict = np.ndarray(shape=(n_points, n_features), dtype=np.float64)
        

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
            neighbour_dict[n_, 0] = 1.0
            # o1
            neighbour_dict[n_, 1] = Eta
            neighbour_dict[n_, 2] = Tau
            # o2
            neighbour_dict[n_, 3] = 0.5*Eta*Eta #
            neighbour_dict[n_, 4] = Eta*Tau
            neighbour_dict[n_, 5] = 0.5*Tau*Tau #
            # o3
            neighbour_dict[n_, 6] = Eta*Eta*Eta
            neighbour_dict[n_, 7] = Eta*Eta*Tau
            neighbour_dict[n_, 8] = Eta*Tau*Tau
            neighbour_dict[n_, 9] = Tau*Tau*Tau
            # o4
            neighbour_dict[n_, 10] = Eta*Eta*Eta*Eta
            neighbour_dict[n_, 11] = Eta*Eta*Eta*Tau
            neighbour_dict[n_, 12] = Eta*Eta*Tau*Tau
            neighbour_dict[n_, 13] = Eta*Tau*Tau*Tau
            neighbour_dict[n_, 14] = Tau*Tau*Tau*Tau
            
        
        derivs = np.dot(np.linalg.pinv(neighbour_dict), neighbour_jump_u)
        self.irr_jump_u_nn[index] = self.irr_jump_f[index]  \
        - self.irr_Kappa[index] * self.irr_jump_u_n[index]  \
        - derivs[3] - derivs[5]

        # TODO

    def __irregular_projection_corr(self, index, i, j, k):
        """ A module for computing corrections.

        Args:
            i, j, k     (integer):      The index of coords of irregular point.

        Returns:
            None

        Computation:
            correction term [u] + d*[u_n] + 1/2 d^2*[u_{nn}].

        """

        d = self.irr_dist[index]
        corr = self.irr_jump_u[index]   \
        + d * self.irr_jump_u_n[index]  \
        #+ 0.5*d*d * self.irr_jump_u_nn[index]

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
            30 + self.pde.mesh.n_x + self.pde.mesh.n_y + 5*self.pde.mesh.n_z    \
            + np.max([self.pde.mesh.n_x, self.pde.mesh.n_y, self.pde.mesh.n_z]) \
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


    def __phi_derivs1_trilinear(self, x, y, z, i, j, k):
        """ A module for estimating the 1st-order derivatives of off-grid points.

        Args:
            x, y, z     (real):     The coords of point.
            i, j, k     (integer):  The floor index of point.

        Returns:
            All 3 1st-order derivatives.

        """

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
        """ A module for estimating the 2nd-order derivatives of off-grid points.

        Args:
            x, y, z     (real):     The coords of point.
            i, j, k     (integer):  The floor index of point.

        Returns:
            All 9 2nd-order derivatives.

        """
        
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

mesh = mesh_uniform(multiplier=1)
inte = interface_ellipsoid(0.6, 0.5, 0.4, mesh)
a = poisson_scc(inte, mesh)
scc = poisson_IIM_solver(a)