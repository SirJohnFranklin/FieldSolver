from __future__ import division, print_function

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.constants as const

from traits.api import HasTraits, on_trait_change, Instance, Bool, Float, Int, Array, List, Dict
from HelperFunctions import plot_field, timeitNV, calculate_radius
from scipy.sparse.linalg import spsolve, bicgstab, bicg, cg, cgs, gmres, lgmres, minres, qmr, lsqr, lsmr
# from scikits.umfpack import spsolve
from scipy import sparse
from tqdm import tqdm

from pyamg import ruge_stuben_solver

# maybe use pysparse to solve matrices

@timeitNV
@nb.jit(nopython=True, nogil=True)
def solve_gauss_seidel_cylindric(initial_phi, cell_type, rho_i, dr, dz, r, maxit=1e5, rtol=1e-6):
    nz, nr = cell_type.shape
    g = np.zeros(cell_type.shape)
    phi = np.zeros(cell_type.shape)
    phib4 = np.zeros(cell_type.shape)
    tol = 0.
    for it in xrange(int(maxit)):
        phib4 = phi
        for i in range(1, nz - 1):  # loop over cells
            for j in range(1, nr - 1):
                b = (rho_i[i, j]) / const.epsilon_0
                g[i, j] = (b +
                           (phi[i, j - 1] + phi[i, j + 1]) / dr**2 +
                           (-phi[i, j - 1] + phi[i, j + 1]) / (2 * dr * r[i,j]) +  # -/+ error in phi part???
                           (phi[i - 1, j] + phi[i + 1, j]) / dz**2) / (2 / dr**2 + 2 / dz**2)

        # neumann boundaries around the "World"
        g[0, :] = g[1, :]  # left
        g[-1, :] = g[-2, :]  # right
        g[:, -1] = g[:, -2]  # top
        g[:, 0] = g[:, 1]  # buttom

        # dirichlet nodes
        phi = np.where(cell_type > 0, initial_phi, g)
        tol = np.nansum(np.abs((phi - phib4)/(phi + phib4)))

        if tol < rtol or it == maxit:
            return phi, tol, it

    return phi, tol, it


@timeitNV
@nb.jit(nopython=True, nogil=True)
def solve_gauss_seidel_cartesian(initial_phi, cell_type, rho_i, dy, dx, maxit=1e5, rtol=1e-6):
    nx, ny = cell_type.shape
    g = np.zeros(cell_type.shape)
    phi = np.zeros(cell_type.shape)
    phib4 = np.zeros(cell_type.shape)
    tol = 0.
    for it in xrange(int(maxit)):
        phib4 = phi
        for i in range(1, nx - 1):  # loop over cells
            for j in range(1, ny - 1):
                b = (rho_i[i, j]) / const.epsilon_0
                g[i, j] = ((phi[i-1,j]+phi[i+1,j])*dy**2 + dx**2 * (phi[i,j-1]+phi[i,j+1] + dy**2 * b)) / (2 * (dx**2 + dy**2))

        # neumann boundaries around the "World"
        g[0, :] = g[1, :]  # left
        g[-1, :] = g[-2, :]  # right
        g[:, -1] = g[:, -2]  # top
        g[:, 0] = g[:, 1]  # buttom

        # dirichlet nodes
        phi = np.where(cell_type > 0, initial_phi, g)
        tol = np.nansum(np.abs((phi - phib4)/(phi + phib4)))

        if tol < rtol or it == maxit:
            return phi, tol, it

    return phi, tol, it


class PoissonSolverBase(HasTraits):
    verbose = Bool(True)

    node_volume = Array
    numbers_to_voltage_dict = Dict
    numbers_to_currect_dict = Dict
    rho_i = Array

    electric_cell_type = Array
    electric_sparse_mat = Instance(sparse.csr_matrix)
    initial_electric_potential = Array
    solved_electric_potential = Array

    magnetic_cell_type = Array
    magnetic_sparse_mat = Instance(sparse.csr_matrix)
    initial_currents = Array
    solved_magnetic_potential = Array

    electric_field = Array
    magnetic_field = Array


    def get_electric_cell_type(self):
        return self.electric_cell_type

    def get_magnetic_cell_type(self):
        return self.magnetic_cell_type

    def set_electric_cell_type(self, cell_type_array, numbers_to_voltage_dict):
        """
        Creates initial potential array from celltypes and voltages assigned to these cell types
        :param cell_type_array: array of int..
        :param numbers_to_voltage_dict: dictionary of number in cell type array to initial voltage
        :return: initial_electric_potential (not solved)
        """
        if self.verbose:
            print(self.__class__.__name__, ": setting electric_cell_type array | electric_cell_type.shape = ", self.electric_cell_type.shape)
        self.electric_cell_type = cell_type_array
        self.numbers_to_voltage_dict = numbers_to_voltage_dict
        for key in numbers_to_voltage_dict:
            num = int(key)
            self.initial_electric_potential[np.where(self.electric_cell_type == num)] = numbers_to_voltage_dict[key]

        return self.initial_electric_potential

    def set_magnetic_cell_type(self, cell_type_array, numbers_to_currents_dict):
        """
        Creates initial potential array from celltypes and voltages assigned to these cell types
        :param cell_type_array: array of int..
        :param numbers_to_voltage_dict: dictionary of number in cell type array to initial voltage
        :return: initial_electric_potential (not solved)
        """
        if self.verbose:
            print(self.__class__.__name__, ": setting magnetic_cell_type array | magnetic_cell_type.shape = ", self.electric_cell_type.shape)
        self.magnetic_cell_type = cell_type_array
        self.numbers_to_current_dict = numbers_to_currents_dict
        for key in numbers_to_currents_dict:
            num = int(key)
            self.initial_currents[np.where(self.magnetic_cell_type == num)] = numbers_to_currents_dict[key]

        return self.initial_currents

    def _get_borders_array(self):
        # this is an helper function for create_Ab_matrix to know where we have the boundaries in the flattened array
        borders = np.zeros(self.electric_cell_type.shape)
        borders[0, :] = 1  # buttom
        borders[-1, :] = 2  # top
        borders[:, 0] = 3  # left
        borders[:, -1] = 4  # right
        ncells = self._get_mesh_size()
        return borders.reshape(ncells[0] * ncells[1])

    def _add_for_COO_format(self, i, j, v):
        self._ii.append(i)
        self._jj.append(j)
        self._va.append(v)

    def _create_electric_rhs(self):
        raise NotImplementedError("Has to be implemented by inheriting classes.")

    def _create_magnetic_rhs(self):
        raise NotImplementedError("Has to be implemented by inheriting classes.")

    @timeitNV
    def _pyamg_benchmark(self, matrix, rhs, x0=None, tol=1e-8):
        ml = ruge_stuben_solver(matrix)
        print(ml)  # print hierarchy information
        x = ml.solve(rhs, x0=x0, tol=tol)  # solve Ax=b to a tolerance of 1e-8
        print(self.__class__.__name__,": Residual norm after direct solving is", np.linalg.norm(rhs - matrix * x))  # compute norm of residual vector
        return x

    def calculate_potential_exact(self, method='iterative'):
        if not np.all(self.initial_currents == 0.):
            self.calculate_electric_magnetic_potential_exact(kind='magnetic', method=method)

        if not np.all(self.initial_electric_potential == 0.):
            self.calculate_electric_magnetic_potential_exact(kind='electric', method=method)


    def calculate_electric_magnetic_potential_exact(self, kind, method='iterative'):
        # TODO: without x0, pyamg finds no solution. Could be used for iterative solving (faster, because multilevel_solver),
        # https://github.com/pyamg/pyamg

        """
        If there is no Ab matrix, function will calculate it by direct solving (scipy.spsolve).
        If there is Ab matrix, programme will use an iterative solver with previously solved potential as initial guess
        by default to speed up computation time.
        :param method: 'iterative' or 'direct'. Only important if sparse matrix has been calculated. Normally iterative
                        is faster. If strong fluctuations in potential occur, e.g. through charged ions/electrons,
                        direct solving is faster.
        :return: solved_electric_potential
        """

        if kind == 'electric':
            self.electric_rhs = self._create_electric_rhs()
            rhs = self.electric_rhs
            sparse_matrix = self.electric_sparse_mat
            x0 = self.solved_electric_potential.reshape(self._get_mesh_size())
        elif kind == 'magnetic':
            self.magnetic_rhs = self._create_magnetic_rhs()
            rhs = self.magnetic_rhs
            sparse_matrix = self.magnetic_sparse_mat
            x0 = self.solved_magnetic_potential.reshape(self._get_mesh_size())
        else:
            raise NotImplementedError(self.__class__.__name__,": kind must be 'both' (=None) or 'electric' or 'magnetic'")

        if sparse_matrix is None:
            sparse_matrix = self.create_Ab_matrix(kind=kind)  # if cell types do not change, this is constant!
            if self.verbose:
                print(self.__class__.__name__,": Solving ", sparse_matrix.shape, " sparse matrix system...")
            x = self._spsolve_benchmark(sparse_matrix, rhs)
            # x = self._pyamg_benchmark(sparse_matrix, rhs)
            if self.verbose:
                print(self.__class__.__name__, ": Residual norm after direct solving is", np.linalg.norm(rhs - sparse_matrix * x))
        else:
            if method == 'iterative':
                # x = self._bicgstab_benchmark(sparse_matrix, rhs, x0=x0)
                x = self._pyamg_benchmark(sparse_matrix, rhs, x0=x0)
            elif method == 'direct':
                x = self._spsolve_benchmark(sparse_matrix, rhs)
            else:
                # print(self.__class__.__name__,": Methods for solving can only be iterative and direct!")
                raise NotImplementedError(self.__class__.__name__,": Methods for solving can only be iterative and direct!")

            if self.verbose:
                print(self.__class__.__name__, ": Residual norm after ", method, " solving is", np.linalg.norm(rhs - sparse_matrix * x))

        if kind == 'electric':
            self.solved_electric_potential = x.reshape(self._get_mesh_size())  # correct
            self.electric_sparse_mat = sparse_matrix
        elif kind == 'magnetic':
            self.solved_magnetic_potential = x.reshape(self._get_mesh_size())
            self.magnetic_sparse_mat = sparse_matrix
        else:
            print("Never happens.")

        return self.solved_electric_potential


    @timeitNV
    def _spsolve_benchmark(self, sparse_matrix, rhs):
        return spsolve(sparse_matrix, rhs)

    def _bicgstab_benchmark(self, sparse_matrix, rhs, x0=None):
        x = bicgstab(sparse_matrix, rhs, x0, tol=1e-5)
        return x[0]

    def _get_mesh_size(self):
        if hasattr(self, 'nr') and hasattr(self, 'nz'):
            return self.nr, self.nz
        elif hasattr(self, 'ny') and hasattr(self, 'nx'):
            return self.ny, self.nx

    def _electric_field_default(self):
        return np.zeros(self._get_mesh_size())

    def _solved_electric_potential_default(self):
        return np.zeros(self._get_mesh_size())

    def _initial_currents_default(self):
        return np.zeros(self._get_mesh_size())

    def _initial_electric_potential_default(self):
        return np.zeros(self._get_mesh_size())

    def _initial_magnetic_potential_default(self):
        return np.zeros(self._get_mesh_size())

    def _rho_i_default(self):
        return np.zeros(self._get_mesh_size())

    def _electric_cell_type_default(self):
        return np.zeros(self._get_mesh_size())

    def _magnetic_cell_type_default(self):
        return np.zeros(self._get_mesh_size())

    def _solved_magnetic_potential_default(self):
        return np.zeros(self._get_mesh_size())

    def _magnetic_field_default(self):
        return np.zeros(self._get_mesh_size())


class CartesianPoissonSolver(PoissonSolverBase):
    """
    - All arrays go like [y,x]
    """
    nx = Int
    ny = Int
    dx = Float
    dy = Float
    xvals = Array
    yvals = Array

    electric_field_x = Array
    electric_field_y = Array

    def __init__(self, nx, dx, ny, dy):
        super(CartesianPoissonSolver, self).__init__()
        print(self.__class__.__name__, ": Created with nx = ", nx, " | dx = ", dx, " | ny = ", ny, " | dy = ", dy)
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = dx
        self.dy = dy

    def plot_all_fields(self, figsize=(12,4)):
        savedict = {}
        savedict['x-values'] = self.xvals
        savedict['y-values'] = self.yvals
        if not np.all(self.solved_electric_potential == 0.):
            plot_field(self.xvals, self.yvals, self.solved_electric_potential, self.electric_cell_type, 'potential $\phi$ [V]', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.initial_electric_potential, self.electric_cell_type, 'initial potential $\phi$ [V]', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.electric_field_y, self.electric_cell_type, 'electric field $E_y$ [V/m]', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.electric_field_x, self.electric_cell_type, 'electric field $E_x$ [V/m]', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.electric_field, self.electric_cell_type, 'electric field combinded $E$ [V/m]', lognorm=False, figsize=figsize)
            savedict['solved_electric_potential'] = self.solved_electric_potential
            savedict['initial_electric_potential'] = self.initial_electric_potential
            savedict['electric_field_y'] = self.electric_field_y
            savedict['electric_field_x'] = self.electric_field_x
            savedict['electric_field_combined'] = self.electric_field


        if not np.all(self.solved_magnetic_potential == 0.):
            plot_field(self.xvals, self.yvals, self.solved_magnetic_potential, self.magnetic_cell_type, 'magnetic potential A [Wb/m]', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.initial_currents, self.electric_cell_type, 'initial currents J $[A/m^2]$', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.magnetic_field_y, self.magnetic_cell_type, 'magnetic field $B_y$ [T]', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.magnetic_field_x, self.magnetic_cell_type, 'magnetic field $B_x$ [T]', figsize=figsize)
            plot_field(self.xvals, self.yvals, self.magnetic_field, self.magnetic_cell_type, 'magnetic field combinded $B$ [T]', lognorm=False)
            savedict['solved_magnetic_potential'] = self.solved_magnetic_potential
            savedict['initial_currents'] = self.initial_currents
            savedict['magnetic_field_y'] = self.magnetic_field_y
            savedict['magnetic_field_x'] = self.magnetic_field_x
            savedict['magnetic_field'] = self.magnetic_field

        for fig in plt.get_fignums():
            plt.figure(fig)
            plt.xlabel('x direction [m]')
            plt.ylabel('y direction [m]')

        return savedict


    def calculate_potential_gauss_seidel(self):
        solved_potential, tol, iters = solve_gauss_seidel_cartesian(np.transpose(self.initial_electric_potential),
                                                                    np.transpose(self.electric_cell_type),
                                                                    np.transpose(self.rho_i), self.dy,
                                                                    self.dx, maxit=1e6)
        self.solved_electric_potential = np.transpose(solved_potential)
        print(self.__class__.__name__, ": solve_gauss_seidel_cartesian() reached tolerance of ", tol, " after ", iters, " iterations.")
        return self.solved_electric_potential

    @on_trait_change('solved_electric_potential')
    def calculate_electric_fields(self):
        if self.verbose:
            print(self.__class__.__name__,": Calculating electric fields")
        self.electric_field_x = -np.gradient(self.solved_electric_potential, self.dx, axis=1, edge_order=2)
        self.electric_field_y = -np.gradient(self.solved_electric_potential, self.dy, axis=0, edge_order=2)
        self.electric_field = np.sqrt(self.electric_field_x ** 2 + self.electric_field_y ** 2)
        return self.electric_field, self.electric_field_y, self.electric_field_x

    @on_trait_change('solved_magnetic_potential')
    def calculate_magnetic_fields(self):
        if self.verbose:
            print(self.__class__.__name__,": Calculating magnetic fields")
        self.magnetic_field_x = -np.gradient(self.solved_magnetic_potential, self.dy, axis=0, edge_order=2)
        self.magnetic_field_y = -np.gradient(self.solved_magnetic_potential, self.dx, axis=1, edge_order=2)
        self.magnetic_field = np.sqrt(self.magnetic_field_x**2 + self.magnetic_field_y**2)
        return self.magnetic_field, self.magnetic_field_y, self.magnetic_field_x

    def _create_electric_rhs(self):
        b = self.initial_electric_potential - self.dx * self.dy * self.rho_i / const.epsilon_0  # TODO: check if dx * dy or dx**2
        b = b.reshape(self.nx * self.ny)
        return b

    def _create_magnetic_rhs(self):
        b = -self.initial_currents * const.mu_0
        b = b.reshape(self.nx * self.ny)
        return b

    @timeitNV
    def create_Ab_matrix(self, kind):
        """
        See https://en.wikipedia.org/wiki/Five-point_stencil
        See http://www.sciencedirect.com/science/article/pii/0010465571900476 for algorithm description, e.g. alpha,
        betaj and gammaj
        """
        if kind == 'electric':
            cell_type_flat = self.electric_cell_type.reshape((self.nx * self.ny))
            rb = -1.
        elif kind == 'magnetic':
            cell_type_flat = self.magnetic_cell_type.reshape((self.ny * self.nx))
            rb = 0.
        else:
            print("This never happens.")
            exit()

        borders = self._get_borders_array()
        max_i = self.nx * self.ny
        max_j = self.nx * self.ny

        self._ii = []
        self._jj = []
        self._va = []

        alpha = -2 * (1 + (self.dx / self.dy) ** 2)
        print(self.__class__.__name__, ": Creating sparse matrix for field solving.")
        for i in tqdm(xrange(max_i)):  # first only Neumann RB, dirichlet via electric_cell_type (e.g. set borders to phi=0)...
            j = np.floor(i / self.nx) + 1
            if cell_type_flat[i] > 0 and kind == 'electric':  # fixed to phi
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 1:  # buttom
                self._add_for_COO_format(i, i + self.nx, rb)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 2:  # top
                self._add_for_COO_format(i, i - self.nx, rb)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 3:  # left
                self._add_for_COO_format(i, i + 1, rb)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 4:  # right
                self._add_for_COO_format(i, i - 1, rb)
                self._add_for_COO_format(i, i, 1.)
            else:
                betaj = (self.dx / self.dy) ** 2
                gammaj = (self.dx / self.dy) ** 2
                if i - 1 >= 0:
                    self._add_for_COO_format(i, i - 1,  1.)
                if i + 1 < max_j:
                    self._add_for_COO_format(i, i + 1, 1.)
                if (i - self.nx) >= 0:  # buttom
                    self._add_for_COO_format(i, i - self.nx, gammaj)
                if (i + self.nx) <= max_j:  # top
                    self._add_for_COO_format(i, i + self.nx, betaj)

                self._add_for_COO_format(i, i, alpha)

        if kind == 'electric':
            self.electric_sparse_mat = sparse.coo_matrix((self._va, (self._ii, self._jj))).tocsr()  #
            return self.electric_sparse_mat
        elif kind == 'magnetic':
            self.magnetic_sparse_mat = sparse.coo_matrix((self._va, (self._ii, self._jj))).tocsr()  #
            return self.magnetic_sparse_mat
        else:
            print("This never happens.")
            exit(-1)

    def _electric_field_x_default(self):
        return np.zeros(self._get_mesh_size())

    def _electric_field_y_default(self):
        return np.zeros(self._get_mesh_size())

    def _yvals_default(self):
        return np.linspace(0, self.ny, self.ny, endpoint=False) * self.dy

    def _xvals_default(self):
        return np.linspace(0, self.nx, self.nx, endpoint=False) * self.dx


class CylindricalPoissonSolver(PoissonSolverBase):
    """
    - All arrays go like [r,z]
    """
    nz = Int
    nr = Int
    dz = Float
    dr = Float
    zvals = Array
    rvals = Array

    electric_field_z = Array
    electric_field_r = Array

    magnetic_field_z = Array
    magnetic_field_r = Array

    def __init__(self, nz, dz, nr, dr):
        super(CylindricalPoissonSolver, self).__init__()
        print(self.__class__.__name__, ": Created with nx = ", nz, " | dx = ", dz, " | ny = ", nr, " | dy = ", dr)
        self.nz = int(nz)
        self.nr = int(nr)
        self.dz = dz
        self.dr = dr

    def plot_all_fields(self):
        savedict = {}
        savedict['r-values'] = self.rvals
        savedict['z-values'] = self.zvals
        if not np.all(self.solved_electric_potential == 0.):
            print(self.__class__.__name__, ": Plotting electric fields")
            plot_field(self.zvals, self.rvals, self.solved_electric_potential, self.electric_cell_type, 'potential $\phi$ [V]', mask=True)
            plot_field(self.zvals, self.rvals, self.initial_electric_potential, self.electric_cell_type, 'initial potential $\phi$ [V]')
            plot_field(self.zvals, self.rvals, self.electric_field_r, self.electric_cell_type, 'electric field $E_r$ [V/m]', mask=True)
            plot_field(self.zvals, self.rvals, self.electric_field_z, self.electric_cell_type, 'electric field $E_z$ [V/m]', mask=True)
            plot_field(self.zvals, self.rvals, self.electric_field, self.electric_cell_type, 'electric field combinded $E$ [V/m]', lognorm=False, mask=True)
            savedict['solved_electric_potential'] = self.solved_electric_potential
            savedict['initial_electric_potential'] = self.initial_electric_potential
            savedict['electric_field_r'] = self.electric_field_r
            savedict['electric_field_z'] = self.electric_field_z
            savedict['electric_field_combined'] = self.electric_field


        if not np.all(self.solved_magnetic_potential == 0.):
            print(self.__class__.__name__, ": Plotting magnetic fields")
            plot_field(self.zvals, self.rvals, self.solved_magnetic_potential, self.magnetic_cell_type, 'magnetic potential $A$ [Wb/m]')
            plot_field(self.zvals, self.rvals, self.initial_currents, self.electric_cell_type, 'initial currents $\phi$ $[A/m^2]$')
            plot_field(self.zvals, self.rvals, self.magnetic_field_r, self.magnetic_cell_type, 'magnetic field $B_r$ [T]')
            plot_field(self.zvals, self.rvals, self.magnetic_field_z, self.magnetic_cell_type, 'magnetic field $B_z$ [T]')
            plot_field(self.zvals, self.rvals, self.magnetic_field, self.magnetic_cell_type, 'magnetic field combinded $B$ [T]', lognorm=False)
            savedict['solved_magnetic_potential'] = self.solved_magnetic_potential
            savedict['initial_currents'] = self.initial_currents
            savedict['magnetic_field_r'] = self.electric_field_r
            savedict['magnetic_field_z'] = self.electric_field_z
            savedict['magnetic_field'] = self.magnetic_field

        for fig in plt.get_fignums():
            plt.figure(fig)
            plt.xlabel('z direction [m]')
            plt.ylabel('r direction [m]')

        return savedict

    def get_radius_array(self):
        r = np.zeros_like(np.transpose(self.electric_cell_type))  # get radii in not standard order for gauss-seidel
        for i in range(self.nz):
            for j in range(self.nr):
                r[i][j] = calculate_radius(j+1, self.dr)

        return r

    def calculate_potential_gauss_seidel(self):

        r = self.get_radius_array()

        solved_potential, tol, iters = solve_gauss_seidel_cylindric(np.transpose(self.initial_electric_potential),
                                                                    np.transpose(self.electric_cell_type),
                                                                    np.transpose(self.rho_i), self.dr,
                                                                    self.dz, r, maxit=1e6)

        self.solved_electric_potential = np.transpose(solved_potential)
        print(self.__class__.__name__, ": solve_gauss_seidel_cylindric() reached tolerance of ", tol, " after ", iters, " iterations.")
        return self.solved_electric_potential

    @on_trait_change('solved_electric_potential')
    def calculate_electric_fields(self):
        if self.verbose:
            print(self.__class__.__name__,": Calculating electric fields")
        self.electric_field_z = -np.gradient(self.solved_electric_potential, self.dz, axis=1, edge_order=2)
        self.electric_field_r = -np.gradient(self.solved_electric_potential, self.dr, axis=0, edge_order=2)
        self.electric_field = np.sqrt(self.electric_field_z**2 + self.electric_field_r**2)
        return self.electric_field, self.electric_field_r, self.electric_field_z

    @on_trait_change('solved_magnetic_potential')
    def calculate_magnetic_fields(self):
        # See: link.springer.com/content/pdf/10.1007%2F3-540-28812-0_2.pdf - p. 27ff
        if self.verbose:
            print(self.__class__.__name__,": Calculating magnetic fields")

        r = np.transpose(self.get_radius_array())
        self.magnetic_field_z = np.gradient(self.solved_magnetic_potential, self.dr, axis=0, edge_order=2) / (2 * np.pi * np.transpose(self.get_radius_array()))
        self.magnetic_field_r = -np.gradient(self.solved_magnetic_potential, self.dz, axis=1, edge_order=2) / (2 * np.pi * np.transpose(self.get_radius_array()))
        self.magnetic_field = np.sqrt(self.magnetic_field_z**2 + self.magnetic_field_r**2)
        return self.magnetic_field, self.magnetic_field_r, self.magnetic_field_z

    def _create_electric_rhs(self):
        # creates the right hand side of possion equation
        # rho_i is density of ions in a cell (see poisson equation)
        # See http://www.sciencedirect.com/science/article/pii/0010465571900476 for dz**2 factor (p in paper)
        b = self.initial_electric_potential - self.dz ** 2 * self.rho_i / const.epsilon_0
        b = b.reshape(self.nz * self.nr)
        return b

    def _create_magnetic_rhs(self):
        # creates the right hand side of possion equation
        # See: link.springer.com/content/pdf/10.1007%2F3-540-28812-0_2.pdf - p. 27ff
        # self.get_radius_array()*self.dr is a j array
        b = - const.mu_0 * self.initial_currents * self.dz ** 2 * (2 * np.pi * np.transpose(self.get_radius_array()))
        b = b.reshape(self.nz * self.nr)
        return b

    @timeitNV
    def create_Ab_matrix(self, kind):
        """
        See https://en.wikipedia.org/wiki/Five-point_stencil
        See http://www.sciencedirect.com/science/article/pii/0010465571900476 for algorithm description, e.g. alpha,
        betaj and gammaj
        :param kind: 'magnetic or electric'
        :return: Nothing
        """
        if kind == 'electric':
            cell_type_flat = self.electric_cell_type.reshape((self.nz * self.nr))
            rb = -1.  # neumann boundary
            intermed_factor = 1.  # See: link.springer.com/content/pdf/10.1007%2F3-540-28812-0_2.pdf - p. 27ff
        elif kind == 'magnetic':
            cell_type_flat = self.magnetic_cell_type.reshape((self.nz * self.nr))
            rb = 0.  # dirichelet boundary
            intermed_factor = -1.  # See: link.springer.com/content/pdf/10.1007%2F3-540-28812-0_2.pdf - p. 27ff
        else:
            print("This never happens.")
            exit()

        print(self.__class__.__name__, ": Creating sparse matrix for", kind, "field solving.")
        borders = self._get_borders_array()
        max_i = self.nz * self.nr
        max_j = self.nz * self.nr

        self._ii = []
        self._jj = []
        self._va = []

        alpha = -2 * (1 + (self.dz/self.dr)**2)

        for i in tqdm(xrange(max_i)):  # first only Neumann RB, dirichlet via electric_cell_type (e.g. set borders to phi=0)...
            j = np.floor(i / self.nz)+1
            if cell_type_flat[i] > 0 and kind == 'electric':  # fixed to phi
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 1:  # buttom
                self._add_for_COO_format(i, i + self.nz, rb)  # disabled = 0: field gets pushed into domain., -1 for electric potential
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 2:  # top
                self._add_for_COO_format(i, i - self.nz, rb)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 3:  # left
                self._add_for_COO_format(i, i + 1, rb)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 4:  # right
                self._add_for_COO_format(i, i - 1, rb)
                self._add_for_COO_format(i, i, 1.)
            else:
                betaj = (self.dz / self.dr) ** 2 * (1 + 1 / (2 * j))
                gammaj = (self.dz / self.dr) ** 2 * (1 - 1 / (2 * j))
                if i - 1 >= 0:
                    self._add_for_COO_format(i, i - 1,  1.)
                if i + 1 <= max_j:
                    self._add_for_COO_format(i, i + 1, 1.)
                if (i - self.nz) >= 0:  # buttom
                    self._add_for_COO_format(i, i + intermed_factor * self.nz, betaj)
                if (i + self.nz) <= max_j:  # top
                    self._add_for_COO_format(i, i - intermed_factor * self.nz, gammaj)

                self._add_for_COO_format(i, i, alpha)

        if kind == 'electric':
            self.electric_sparse_mat = sparse.coo_matrix((self._va, (self._ii, self._jj))).tocsr()  #
            return self.electric_sparse_mat
        elif kind == 'magnetic':
            self.magnetic_sparse_mat = sparse.coo_matrix((self._va, (self._ii, self._jj))).tocsr()  #
            return self.magnetic_sparse_mat
        else:
            print("This never happens.")
            exit()

    def _electric_field_z_default(self):
        return np.zeros(self._get_mesh_size())

    def _electric_field_r_default(self):
        return np.zeros(self._get_mesh_size())

    def _rvals_default(self):
        return np.linspace(0, self.nr, self.nr, endpoint=False) * self.dr

    def _zvals_default(self):
        return np.linspace(0, self.nz, self.nz, endpoint=False) * self.dz