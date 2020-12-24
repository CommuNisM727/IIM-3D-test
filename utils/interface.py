### ALL REQUIRED PYTHON MODULES.
import numpy as np
import time

from utils.utils_basic import *

""" DEBUG
#from utils_basic import *
from mesh import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""
###-----------------------------------------------------------------------------------
### FILE NAME:      interface.py
### CREATE DATE:    DEC. 2020.
### AUTHOR:         Yuan-Tian (CommuNisM727)
###-----------------------------------------------------------------------------------
### DESCRIPTION:    Indicating the interface through level-set function \phi.
### NOTED:          A simple ellipsoid interface and its inaccurate level-set 
###                 function is given.
###                 More interface may be added later.
###-----------------------------------------------------------------------------------


class interface_ellipsoid(object):
    """ A simple ellipsoid interface (x^2/a^2 + y^2/b^2 + z^2/c^2 = 1).

    Attributes:
        a, b, c     (real):             Elliptic radius along x, y, z.
        irr         (3D-array):         Indices of irregular mesh points [1, n_irr].
        n_irr       (integer):          Number of irregular mesh points.
        phi         (3D-array):         Level-set function \phi computed on mesh points.
        phi_        (3D-array):         1st-order derivatives of \phi. 
        phi__       (3D-array):         2nd-order derivatives of \phi.
        
        irr_proj    (1D*3-array):       An array of projections of irregular points.
        irr_dist    (1D-array):         An array of distances to the interface of irregular points.
        
        irr_Xi      (1D*3-array):       An array of surface normal direction vector \Xi.
        irr_Eta     (1D*3-array):       An array of surface tangential vector \Eta.
        irr_Tau     (1D*3-array):       An array of surface tangential vector \Tau.
        irr_Kappa   (1D-array):         An array of surface mean curveture.

    """
    def __init__(self, a, b, c, mesh):
        """ Initialization of class 'interface_ellipsoid'
            Irrgular mesh points 'irr' and 'n_irr' are computed.
            Level-set function '\phi' and its derivatives '\phi_' are computed.

        Args:
            a, b, c     (real):             Elliptic radius along eigen-direction x, y, z.
            mesh        (mesh_uniform):     An uniform mesh in 3D cartesian coordinates 
                                            indicating the computational area.

        Returns:
            None

        """

        self.a = a
        self.b = b
        self.c = c
        self.mesh = mesh
        self.phi = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.irr = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.int)

        self.phi_x = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_y = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_z = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_xx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_xy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_xz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_yx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_yy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_yz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_zx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_zy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_zz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)

        # STEP 1:  Find the value of level-set function \phi on all mesh points.
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    self.phi[i, j, k] = self.__phi(mesh.xs[i], mesh.ys[j], mesh.zs[k])

        # STEP 2:  Find the derivatives of \phi and map irregular points to a index.
        self.n_irr = 0
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    if (i == 0 or i == mesh.n_x or
                        j == 0 or j == mesh.n_y or
                        k == 0 or k == mesh.n_z):
                        continue

                    # Irregular points must be sought AFTER \phi is calculated.
                    if (self.phi[i, j, k] == 0.0):
                        self.irr[i, j, k] = 1
                    else:
                        # A little trick.
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

                    # Derivatives of \phi.
                    self.phi_x[i, j, k] = self.__phi_x(i, j, k, mesh.h_x)
                    self.phi_y[i, j, k] = self.__phi_y(i, j, k, mesh.h_y)
                    self.phi_z[i, j, k] = self.__phi_z(i, j, k, mesh.h_z)

                    self.phi_xx[i, j, k] = self.__phi_xx(i, j, k, mesh.h_x)
                    self.phi_xy[i, j, k] = self.__phi_xy(i, j, k, mesh.h_x, mesh.h_y)
                    self.phi_xz[i, j, k] = self.__phi_xz(i, j, k, mesh.h_x, mesh.h_z)
                    
                    self.phi_yx[i, j, k] = self.__phi_yx(i, j, k, mesh.h_x, mesh.h_y)
                    self.phi_yy[i, j, k] = self.__phi_yy(i, j, k, mesh.h_y)
                    self.phi_yz[i, j, k] = self.__phi_yz(i, j, k, mesh.h_y, mesh.h_z)
                    
                    self.phi_zx[i, j, k] = self.__phi_zx(i, j, k, mesh.h_x, mesh.h_z)
                    self.phi_zy[i, j, k] = self.__phi_zy(i, j, k, mesh.h_y, mesh.h_z)
                    self.phi_zz[i, j, k] = self.__phi_zz(i, j, k, mesh.h_z)

        # Irregular projections.
        self.irr_proj = np.ndarray(shape=(self.n_irr + 1, 3), dtype=np.float64)
        self.irr_dist = np.ndarray(shape=(self.n_irr + 1, ), dtype=np.float64)
        self.irr_Xi = np.ndarray(shape=(self.n_irr + 1, 3), dtype=np.float64)
        self.irr_Eta = np.ndarray(shape=(self.n_irr + 1, 3), dtype=np.float64)
        self.irr_Tau = np.ndarray(shape=(self.n_irr + 1, 3), dtype=np.float64)
        self.irr_Kappa = np.ndarray(shape=(self.n_irr + 1, ), dtype=np.float64)

        # STEP 3:   Find the irregular projections and interface information.
        self.__irregular_projection()

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
        Args & Returns:
            None

        """

        time_a=time.time()
        for i in range(1, self.mesh.n_x):
            for j in range(1, self.mesh.n_y):
                for k in range(1, self.mesh.n_z):
                    if (self.irr[i, j, k] > 0):
                        index = self.irr[i, j, k]
                        # (1). Initialize the projection.
                        [alpha, x_0, y_0, z_0] = self.__irregular_projection_initial(i, j, k)
                        # (2). Apply several Newton iterations to obtain the final projection.
                        [alpha, x_p, y_p, z_p] = self.__irregular_projection_iterate(i, j, k, x_0, y_0, z_0, alpha)
                        # (3). Calculate the interface structures.
                        self.__irregular_projection_info(index, x_p, y_p, z_p, alpha)
        time_b=time.time()
        print('T_proj_curve: ', time_b-time_a)


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
        phi_x = self.phi_x[i, j, k]
        phi_y = self.phi_y[i, j, k]   
        phi_z = self.phi_z[i, j, k]           
        norm_grad_phi = np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)
        # Normalize the gradient.
        phi_x = phi_x / norm_grad_phi
        phi_y = phi_y / norm_grad_phi
        phi_z = phi_z / norm_grad_phi

        phi_xx = self.phi_xx[i, j, k]
        phi_xy = self.phi_xy[i, j, k]
        phi_xz = self.phi_xz[i, j, k]
        
        phi_yx = self.phi_yx[i, j, k]
        phi_yy = self.phi_yy[i, j, k]
        phi_yz = self.phi_yz[i, j, k]
        
        phi_zx = self.phi_zx[i, j, k]
        phi_zy = self.phi_zy[i, j, k]
        phi_zz = self.phi_zz[i, j, k]
        
        # Quadratic form (IIM overview [Li], CLAIMED to have 3rd ACC).
        # \phi + |\nabla \phi| * \alpha +  \alpha^2/2 * (p^t Hess p) = 0
        a = phi_x * (phi_xx*phi_x + phi_xy*phi_y + phi_xz*phi_z)    \
        +   phi_y * (phi_yx*phi_x + phi_yy*phi_y + phi_yz*phi_z)    \
        +   phi_z * (phi_zx*phi_x + phi_zy*phi_y + phi_zz*phi_z)
        b = norm_grad_phi
        c = self.phi[i, j, k]

        #print("point: ", self.mesh.xs[i], self.mesh.ys[j], self.mesh.zs[k])

        [r1, r2] = root_p2(0.5 * a, b, c)
        if (np.abs(r1) <= np.abs(r2)):
            return [r1, self.mesh.xs[i] + r1 * phi_x,
                        self.mesh.ys[j] + r1 * phi_y,
                        self.mesh.zs[k] + r1 * phi_z]
        else:
            return [r2, self.mesh.xs[i] + r2 * phi_x,
                        self.mesh.ys[j] + r2 * phi_y,
                        self.mesh.zs[k] + r2 * phi_z]

    def __irregular_projection_iterate(self, i_0, j_0, k_0, x_0, y_0, z_0, alpha, iter_max=10, eps=1e-10):
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
        x_fix = self.mesh.xs[i_0]
        y_fix = self.mesh.ys[j_0]
        z_fix = self.mesh.zs[k_0]
        
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
        i = int((x_k - self.mesh.x_inf) / self.mesh.h_x)
        j = int((y_k - self.mesh.y_inf) / self.mesh.h_y)
        k = int((z_k - self.mesh.z_inf) / self.mesh.h_z)

        Pn_interp_3d_partial = lambda arr, n: Pn_interp_3d(x_k, y_k, z_k, 
                                                self.mesh.xs, self.mesh.ys, self.mesh.zs,
                                                i - int((n - 1) / 2), j - int((n - 1) / 2), k - int((n - 1) / 2),
                                                arr, n)
        # Tricubic interpolating (4^3 points).
        phi = Pn_interp_3d_partial(self.phi, 4)

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

        i = int((x - self.mesh.x_inf) / self.mesh.h_x)
        j = int((y - self.mesh.y_inf) / self.mesh.h_y)
        k = int((z - self.mesh.z_inf) / self.mesh.h_z)

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

        i = int((x - self.mesh.x_inf) / self.mesh.h_x)
        j = int((y - self.mesh.y_inf) / self.mesh.h_y)
        k = int((z - self.mesh.z_inf) / self.mesh.h_z)

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

    def __phi(self, x, y, z):
        """ Level-set function \phi on one point are computed.
            This implementation is an INACCURATE one based on the analytical equation of 
            ellipsoid, more accurate result can be obtained if a Signed-Distance-Function
            is calculated through iterative methods or so.

        Args:
            x, y, z     (real):     A Triplet indicating the point (x, y, z).

        Returns:
            \phi(x, y, z) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1.
        """
        return x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2 - 1.0

    """ All derivatives (central difference) of level-set function \phi on one 
        point are computed, private method called by the initialization function.

    Args:
        i, j, k         (integer):      A Triplet indicating the point (x_i, y_j, z_k).
        h_x, h_y, h_z   (real):         Distance between points along x, y, z axis.

    Returns:
        Central difference approximation of all derivatives of \phi.
    """
    # 1st-order.
    def __phi_x(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] - self.phi[i - 1, j, k]) / (2.0 * h_x)
    def __phi_y(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] - self.phi[i, j - 1, k]) / (2.0 * h_y)
    def __phi_z(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] - self.phi[i, j, k - 1]) / (2.0 * h_z)
    # 2nd-order.
    def __phi_xx(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] + self.phi[i - 1, j, k] - 2.0 * self.phi[i, j, k]) / (h_x * h_x)
    def __phi_xy(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def __phi_xz(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    
    def __phi_yx(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def __phi_yy(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] + self.phi[i, j - 1, k] - 2.0 * self.phi[i, j, k]) / (h_y * h_y)
    def __phi_yz(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    
    def __phi_zx(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    def __phi_zy(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    def __phi_zz(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] + self.phi[i, j, k - 1] - 2.0 * self.phi[i, j, k]) / (h_z * h_z)


    def __phi_derivs1_trilinear(self, x, y, z, i, j, k):
        """ A module for estimating the 1st-order derivatives of off-grid points.

        Args:
            x, y, z     (real):     The coords of point.
            i, j, k     (integer):  The floor index of point.

        Returns:
            All 3 1st-order derivatives.

        """

        P2_interp_3d = lambda arr: Pn_interp_3d(x, y, z, 
                                                self.mesh.xs, self.mesh.ys, self.mesh.zs,
                                                i + 0, j + 0, k + 0,
                                                arr, 2)

        # Trilinear interpolating.
        phi_x = P2_interp_3d(self.phi_x)
        phi_y = P2_interp_3d(self.phi_y)
        phi_z = P2_interp_3d(self.phi_z)

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
                                                self.mesh.xs, self.mesh.ys, self.mesh.zs,
                                                i + 0, j + 0, k + 0,
                                                arr, 2)

        phi_xx = P2_interp_3d(self.phi_xx)
        phi_xy = P2_interp_3d(self.phi_xy)
        phi_xz = P2_interp_3d(self.phi_xz)
        
        phi_yx = P2_interp_3d(self.phi_yx)
        phi_yy = P2_interp_3d(self.phi_yy)
        phi_yz = P2_interp_3d(self.phi_yz)
        
        phi_zx = P2_interp_3d(self.phi_zx)
        phi_zy = P2_interp_3d(self.phi_zy)
        phi_zz = P2_interp_3d(self.phi_zz)
        
        return [phi_xx, phi_xy, phi_xz, phi_yx, phi_yy, phi_yz, phi_zx, phi_zy, phi_zz]


class interface_ellipsoid_aug(object):
    """ A simple ellipsoid interface (x^2/a^2 + y^2/b^2 + z^2/c^2 = 1).

    Attributes:
        a, b, c     (real):             Elliptic radius along x, y, z.
        irr         (3D-array):         Indices of irregular mesh points [1, n_irr].
        n_irr       (integer):          Number of irregular mesh points.
        phi         (3D-array):         Level-set function \phi computed on mesh points.
        phi_        (3D-array):         1st-order derivatives of \phi. 
        phi__       (3D-array):         2nd-order derivatives of \phi.
        
        irr_proj    (1D*3-array):       An array of projections of irregular points.
        irr_dist    (1D-array):         An array of distances to the interface of irregular points.
        
        irr_Xi      (1D*3-array):       An array of surface normal direction vector \Xi.
        irr_Eta     (1D*3-array):       An array of surface tangential vector \Eta.
        irr_Tau     (1D*3-array):       An array of surface tangential vector \Tau.
        irr_Kappa   (1D-array):         An array of surface mean curveture.

    """
    def __init__(self, a, b, c, mesh):
        """ Initialization of class 'interface_ellipsoid'
            Irrgular mesh points 'irr' and 'n_irr' are computed.
            Level-set function '\phi' and its derivatives '\phi_' are computed.

        Args:
            a, b, c     (real):             Elliptic radius along eigen-direction x, y, z.
            mesh        (mesh_uniform):     An uniform mesh in 3D cartesian coordinates 
                                            indicating the computational area.

        Returns:
            None

        """

        self.a = a
        self.b = b
        self.c = c
        self.mesh = mesh
        self.phi = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.irr = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.int)

        self.phi_x = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_y = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_z = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_xx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_xy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_xz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_yx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_yy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_yz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        
        self.phi_zx = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_zy = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)
        self.phi_zz = np.zeros((mesh.n_x + 1, mesh.n_y + 1, mesh.n_z + 1), dtype=np.float64)

        # STEP 1:  Find the value of level-set function \phi on all mesh points.
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    self.phi[i, j, k] = self.__phi(mesh.xs[i], mesh.ys[j], mesh.zs[k])

        # STEP 2:  Find the derivatives of \phi and map irregular points to a index.
        self.n_irr = 0
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    if (i == 0 or i == mesh.n_x or
                        j == 0 or j == mesh.n_y or
                        k == 0 or k == mesh.n_z):
                        continue

                    # Irregular points must be sought AFTER \phi is calculated.
                    if (self.phi[i, j, k] == 0.0):
                        self.irr[i, j, k] = 1
                    else:
                        # A little trick.
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

                    # Derivatives of \phi.
                    self.phi_x[i, j, k] = self.__phi_x(i, j, k, mesh.h_x)
                    self.phi_y[i, j, k] = self.__phi_y(i, j, k, mesh.h_y)
                    self.phi_z[i, j, k] = self.__phi_z(i, j, k, mesh.h_z)

                    self.phi_xx[i, j, k] = self.__phi_xx(i, j, k, mesh.h_x)
                    self.phi_xy[i, j, k] = self.__phi_xy(i, j, k, mesh.h_x, mesh.h_y)
                    self.phi_xz[i, j, k] = self.__phi_xz(i, j, k, mesh.h_x, mesh.h_z)
                    
                    self.phi_yx[i, j, k] = self.__phi_yx(i, j, k, mesh.h_x, mesh.h_y)
                    self.phi_yy[i, j, k] = self.__phi_yy(i, j, k, mesh.h_y)
                    self.phi_yz[i, j, k] = self.__phi_yz(i, j, k, mesh.h_y, mesh.h_z)
                    
                    self.phi_zx[i, j, k] = self.__phi_zx(i, j, k, mesh.h_x, mesh.h_z)
                    self.phi_zy[i, j, k] = self.__phi_zy(i, j, k, mesh.h_y, mesh.h_z)
                    self.phi_zz[i, j, k] = self.__phi_zz(i, j, k, mesh.h_z)

        # Irregular projections.
        self.irr_proj = np.ndarray(shape=(2 * self.n_irr + 1, 3), dtype=np.float64)
        self.irr_dist = np.ndarray(shape=(2 * self.n_irr + 1, ), dtype=np.float64)
        self.irr_Xi = np.ndarray(shape=(2 * self.n_irr + 1, 3), dtype=np.float64)
        self.irr_Eta = np.ndarray(shape=(2 * self.n_irr + 1, 3), dtype=np.float64)
        self.irr_Tau = np.ndarray(shape=(2 * self.n_irr + 1, 3), dtype=np.float64)
        self.irr_Kappa = np.ndarray(shape=(2 * self.n_irr + 1, ), dtype=np.float64)

        # STEP 3:   Find the irregular projections and interface information.
        self.__irregular_projection()
        
        tmp = self.n_irr
        for i in range(mesh.n_x + 1):
            for j in range(mesh.n_y + 1):
                for k in range(mesh.n_z + 1):
                    if (self.irr[i, j, k] > 0):
                        index = self.irr[i, j, k]
                        x = self.irr_proj[index, 0]
                        y = self.irr_proj[index, 1]
                        z = self.irr_proj[index, 2]
                        i0 = int((x - self.mesh.x_inf) / self.mesh.h_x)
                        j0 = int((y - self.mesh.y_inf) / self.mesh.h_y)
                        k0 = int((z - self.mesh.z_inf) / self.mesh.h_z)
                        
                        for i1 in range(2):
                            for j1 in range(2):
                                for k1 in range(2):
                                    if (self.irr[i0+i1, j0+j1, k0+k1] == 0 and self.phi[i0+i1, j0+j1, k0+k1] > 0):
                                        self.n_irr = self.n_irr + 1
                                        self.irr[i0+i1, j0+j1, k0+k1] = -self.n_irr

        self.__irregular_projection(-1)
        self.n_app = self.n_irr - tmp
        self.n_irr = tmp
        print('irr: ', self.n_irr, 'app: ', self.n_app)

    def __irregular_projection(self, sign=1):
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
        Args & Returns:
            None

        """

        time_a=time.time()
        for i in range(1, self.mesh.n_x):
            for j in range(1, self.mesh.n_y):
                for k in range(1, self.mesh.n_z):
                    if (sign * self.irr[i, j, k] > 0):
                        index = sign * self.irr[i, j, k]
                        # (1). Initialize the projection.
                        [alpha, x_0, y_0, z_0] = self.__irregular_projection_initial(i, j, k)
                        # (2). Apply several Newton iterations to obtain the final projection.
                        [alpha, x_p, y_p, z_p] = self.__irregular_projection_iterate(i, j, k, x_0, y_0, z_0, alpha)
                        # (3). Calculate the interface structures.
                        self.__irregular_projection_info(index, x_p, y_p, z_p, alpha)
        time_b=time.time()
        print('T_proj_curve: ', time_b-time_a)


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
        phi_x = self.phi_x[i, j, k]
        phi_y = self.phi_y[i, j, k]   
        phi_z = self.phi_z[i, j, k]           
        norm_grad_phi = np.sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z)
        # Normalize the gradient.
        phi_x = phi_x / norm_grad_phi
        phi_y = phi_y / norm_grad_phi
        phi_z = phi_z / norm_grad_phi

        phi_xx = self.phi_xx[i, j, k]
        phi_xy = self.phi_xy[i, j, k]
        phi_xz = self.phi_xz[i, j, k]
        
        phi_yx = self.phi_yx[i, j, k]
        phi_yy = self.phi_yy[i, j, k]
        phi_yz = self.phi_yz[i, j, k]
        
        phi_zx = self.phi_zx[i, j, k]
        phi_zy = self.phi_zy[i, j, k]
        phi_zz = self.phi_zz[i, j, k]
        
        # Quadratic form (IIM overview [Li], CLAIMED to have 3rd ACC).
        # \phi + |\nabla \phi| * \alpha +  \alpha^2/2 * (p^t Hess p) = 0
        a = phi_x * (phi_xx*phi_x + phi_xy*phi_y + phi_xz*phi_z)    \
        +   phi_y * (phi_yx*phi_x + phi_yy*phi_y + phi_yz*phi_z)    \
        +   phi_z * (phi_zx*phi_x + phi_zy*phi_y + phi_zz*phi_z)
        b = norm_grad_phi
        c = self.phi[i, j, k]

        #print("point: ", self.mesh.xs[i], self.mesh.ys[j], self.mesh.zs[k])

        [r1, r2] = root_p2(0.5 * a, b, c)
        if (np.abs(r1) <= np.abs(r2)):
            return [r1, self.mesh.xs[i] + r1 * phi_x,
                        self.mesh.ys[j] + r1 * phi_y,
                        self.mesh.zs[k] + r1 * phi_z]
        else:
            return [r2, self.mesh.xs[i] + r2 * phi_x,
                        self.mesh.ys[j] + r2 * phi_y,
                        self.mesh.zs[k] + r2 * phi_z]

    def __irregular_projection_iterate(self, i_0, j_0, k_0, x_0, y_0, z_0, alpha, iter_max=10, eps=1e-10):
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
        x_fix = self.mesh.xs[i_0]
        y_fix = self.mesh.ys[j_0]
        z_fix = self.mesh.zs[k_0]
        
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
        i = int((x_k - self.mesh.x_inf) / self.mesh.h_x)
        j = int((y_k - self.mesh.y_inf) / self.mesh.h_y)
        k = int((z_k - self.mesh.z_inf) / self.mesh.h_z)

        Pn_interp_3d_partial = lambda arr, n: Pn_interp_3d(x_k, y_k, z_k, 
                                                self.mesh.xs, self.mesh.ys, self.mesh.zs,
                                                i - int((n - 1) / 2), j - int((n - 1) / 2), k - int((n - 1) / 2),
                                                arr, n)
        # Tricubic interpolating (4^3 points).
        phi = Pn_interp_3d_partial(self.phi, 4)

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

        i = int((x - self.mesh.x_inf) / self.mesh.h_x)
        j = int((y - self.mesh.y_inf) / self.mesh.h_y)
        k = int((z - self.mesh.z_inf) / self.mesh.h_z)

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

        i = int((x - self.mesh.x_inf) / self.mesh.h_x)
        j = int((y - self.mesh.y_inf) / self.mesh.h_y)
        k = int((z - self.mesh.z_inf) / self.mesh.h_z)

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

    def __phi(self, x, y, z):
        """ Level-set function \phi on one point are computed.
            This implementation is an INACCURATE one based on the analytical equation of 
            ellipsoid, more accurate result can be obtained if a Signed-Distance-Function
            is calculated through iterative methods or so.

        Args:
            x, y, z     (real):     A Triplet indicating the point (x, y, z).

        Returns:
            \phi(x, y, z) = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1.
        """
        return x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2 - 1.0

    """ All derivatives (central difference) of level-set function \phi on one 
        point are computed, private method called by the initialization function.

    Args:
        i, j, k         (integer):      A Triplet indicating the point (x_i, y_j, z_k).
        h_x, h_y, h_z   (real):         Distance between points along x, y, z axis.

    Returns:
        Central difference approximation of all derivatives of \phi.
    """
    # 1st-order.
    def __phi_x(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] - self.phi[i - 1, j, k]) / (2.0 * h_x)
    def __phi_y(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] - self.phi[i, j - 1, k]) / (2.0 * h_y)
    def __phi_z(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] - self.phi[i, j, k - 1]) / (2.0 * h_z)
    # 2nd-order.
    def __phi_xx(self, i, j, k, h_x):
        return (self.phi[i + 1, j, k] + self.phi[i - 1, j, k] - 2.0 * self.phi[i, j, k]) / (h_x * h_x)
    def __phi_xy(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def __phi_xz(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    
    def __phi_yx(self, i, j, k, h_x, h_y):
        return (self.phi[i - 1, j - 1, k] + self.phi[i + 1, j + 1, k] - self.phi[i - 1, j + 1, k] - self.phi[i + 1, j - 1, k]) / (4.0 * h_x * h_y)
    def __phi_yy(self, i, j, k, h_y):
        return (self.phi[i, j + 1, k] + self.phi[i, j - 1, k] - 2.0 * self.phi[i, j, k]) / (h_y * h_y)
    def __phi_yz(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    
    def __phi_zx(self, i, j, k, h_x, h_z):
        return (self.phi[i - 1, j, k - 1] + self.phi[i + 1, j, k + 1] - self.phi[i - 1, j, k + 1] - self.phi[i + 1, j, k - 1]) / (4.0 * h_x * h_z)
    def __phi_zy(self, i, j, k, h_y, h_z):
        return (self.phi[i, j - 1, k - 1] + self.phi[i, j + 1, k + 1] - self.phi[i, j - 1, k + 1] - self.phi[i, j + 1, k - 1]) / (4.0 * h_y * h_z)
    def __phi_zz(self, i, j, k, h_z):
        return (self.phi[i, j, k + 1] + self.phi[i, j, k - 1] - 2.0 * self.phi[i, j, k]) / (h_z * h_z)


    def __phi_derivs1_trilinear(self, x, y, z, i, j, k):
        """ A module for estimating the 1st-order derivatives of off-grid points.

        Args:
            x, y, z     (real):     The coords of point.
            i, j, k     (integer):  The floor index of point.

        Returns:
            All 3 1st-order derivatives.

        """

        P2_interp_3d = lambda arr: Pn_interp_3d(x, y, z, 
                                                self.mesh.xs, self.mesh.ys, self.mesh.zs,
                                                i + 0, j + 0, k + 0,
                                                arr, 2)

        # Trilinear interpolating.
        phi_x = P2_interp_3d(self.phi_x)
        phi_y = P2_interp_3d(self.phi_y)
        phi_z = P2_interp_3d(self.phi_z)

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
                                                self.mesh.xs, self.mesh.ys, self.mesh.zs,
                                                i + 0, j + 0, k + 0,
                                                arr, 2)

        phi_xx = P2_interp_3d(self.phi_xx)
        phi_xy = P2_interp_3d(self.phi_xy)
        phi_xz = P2_interp_3d(self.phi_xz)
        
        phi_yx = P2_interp_3d(self.phi_yx)
        phi_yy = P2_interp_3d(self.phi_yy)
        phi_yz = P2_interp_3d(self.phi_yz)
        
        phi_zx = P2_interp_3d(self.phi_zx)
        phi_zy = P2_interp_3d(self.phi_zy)
        phi_zz = P2_interp_3d(self.phi_zz)
        
        return [phi_xx, phi_xy, phi_xz, phi_yx, phi_yy, phi_yz, phi_zx, phi_zy, phi_zz]

""" MODULE TESTS
mesh = mesh_uniform(multiplier=2)
inte = interface_ellipsoid(0.5, 0.5, 0.2, mesh)

fig = plt.figure()            
ax = fig.gca(projection='3d')

x = []
y = []
z = []
for i in range(mesh.n_x):
    for j in range(mesh.n_y):
        for k in range(mesh.n_z):
            if (inte.irr[i, j, k] > 0):
                x.append(mesh.xs[i])
                y.append(mesh.ys[j])
                z.append(mesh.zs[k])
ax.scatter(x, y, z, linewidth=1)
plt.show()
"""

### MODIFY HISTORY---
### 09.12.2020      FILE CREATED.           ---727
### 22.12.2020      Ver. 0.2 CREATED.       ---727
###-----------------------------------------------------------------------