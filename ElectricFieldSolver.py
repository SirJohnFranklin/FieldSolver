from __future__ import division, print_function

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.constants as const

from traits.api import HasTraits, on_trait_change, Instance, Bool, Float, Int, Array, List, Dict
from HelperFunctions import plot_field, timeitNV, calculate_radius
from scipy.sparse.linalg import spsolve, bicgstab, bicg, cg, cgs, gmres, lgmres, minres, qmr, lsqr, lsmr
from scipy import sparse
from tqdm import tqdm

# maybe use pysparse to solve matrices

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
    sA = Instance(sparse.csr_matrix)
    initial_potential = Array
    cell_type = Array
    rho_i = Array
    solved_potential = Array
    electric_field = Array

    def get_cell_type(self):
        return self.cell_type

    def set_cell_type(self, cell_type_array, numbers_to_voltage_dict):
        """
        Creates initial potential array from celltypes and voltages assigned to these cell types
        :param cell_type_array: array of int..
        :param numbers_to_voltage_dict: dictionary of number in cell type array to initial voltage
        :return: initial_potential (not solved)
        """
        if self.verbose:
            print(self.__class__.__name__, ": setting cell_type array | cell_type.shape = ", self.cell_type.shape)
        self.cell_type = cell_type_array
        self.numbers_to_voltage_dict = numbers_to_voltage_dict
        for key in numbers_to_voltage_dict:
            num = int(key)
            self.initial_potential[np.where(self.cell_type == num)] = numbers_to_voltage_dict[key]

        return self.initial_potential

    @timeitNV
    def _spsolve_benchmark(self):
        return spsolve(self.sA, self.b)

    def _bicgstab_benchmark(self, x0=None):
        x = bicgstab(self.sA, self.b, x0, tol=1e-5)
        return x[0]

    def _get_borders_array(self):
        # this is an helper function for create_Ab_matrix to know where we have the boundaries in the flattened array
        borders = np.zeros(self.cell_type.shape)
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

    def calculate_potential_exact(self, method='iterative'):
        """
        If there is no Ab matrix, function will calculate it by direct solving (scipy.spsolve).
        If there is Ab matrix, programme will use an iterative solver with previously solved potential as initial guess
        by default to speed up computation time.
        :param method: iterative
        :return: solved_potential
        """

        self.b = self._create_b_vector()
        if self.sA is None:
            self.create_Ab_matrix()  # if cell types do not change, this is constant!
            if self.verbose:
                print(self.__class__.__name__,": Solving ", self.sA.shape, " sparse matrix system...")
            x = self._spsolve_benchmark()
        else:
            if method == 'iterative':
                x = self._bicgstab_benchmark(x0=self.solved_potential.reshape(self._get_mesh_size()))
            elif method == 'direct':
                x = self.spsolve_benchmark()
            else:
                print(self.__class__.__name__,": Methods for solving can only be iterative and direct!")
                exit()

        self.solved_potential = x.reshape(self._get_mesh_size())  # correct
        return self.solved_potential

    def _get_mesh_size(self):
        if hasattr(self, 'nr') and hasattr(self, 'nz'):
            return self.nr, self.nz
        elif hasattr(self, 'ny') and hasattr(self, 'nx'):
            return self.ny, self.nx

    def _electric_field_default(self):
        return np.zeros(self._get_mesh_size())

    def _solved_potential_default(self):
        return np.zeros(self._get_mesh_size())

    def _initial_potential_default(self):
        return np.zeros(self._get_mesh_size())

    def _rho_i_default(self):
        return np.zeros(self._get_mesh_size())

    def _cell_type_default(self):
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

    def plot_all_fields(self):
        plot_field(self.xvals, self.yvals, self.solved_potential, self.cell_type, 'potential $\phi$ [V]')
        plot_field(self.xvals, self.yvals, self.initial_potential, self.cell_type, 'initial potential $\phi$ [V]')
        plot_field(self.xvals, self.yvals, self.electric_field_y, self.cell_type, 'electric field $E_r$ [V/m]')
        plot_field(self.xvals, self.yvals, self.electric_field_x, self.cell_type, 'electric field $E_z$ [V/m]')
        plot_field(self.xvals, self.yvals, self.electric_field, self.cell_type, 'electric field combinded $E$ [V/m]', lognorm=False)
        for fig in plt.get_fignums():
            plt.figure(fig)
            plt.xlabel('x direction [m]')
            plt.ylabel('y direction [m]')

    def calculate_potential_gauss_seidel(self):
        solved_potential, tol, iters = solve_gauss_seidel_cartesian(np.transpose(self.initial_potential),
                                                                    np.transpose(self.cell_type),
                                                                    np.transpose(self.rho_i), self.dy,
                                                                    self.dx, maxit=1e6)
        self.solved_potential = np.transpose(solved_potential)
        print(self.__class__.__name__, ": solve_gauss_seidel_cartesian() reached tolerance of ", tol, " after ", iters, " iterations.")
        return self.solved_potential


    @on_trait_change('solved_potential')
    def calculate_electric_fields(self):
        if self.verbose:
            print(self.__class__.__name__,": Calculating electric fields")
        self.electric_field_x = -np.gradient(self.solved_potential, self.dx, axis=1, edge_order=2)
        self.electric_field_y = -np.gradient(self.solved_potential, self.dy, axis=0, edge_order=2)
        self.electric_field = np.sqrt(self.electric_field_x ** 2 + self.electric_field_y ** 2)
        return self.electric_field, self.electric_field_y, self.electric_field_x

    def _create_b_vector(self):
        # creates the right hand side of possion equation
        # rho_i is density of ions in a cell (see poisson equation)
        # See http://www.sciencedirect.com/science/article/pii/0010465571900476 for dx**2 factor (p in paper)
        b = self.initial_potential - self.dx * self.dy * self.rho_i / const.epsilon_0
        b = b.reshape(self.nx * self.ny)
        return b

    @timeitNV
    def create_Ab_matrix(self):
        """
        See https://en.wikipedia.org/wiki/Five-point_stencil
        See http://www.sciencedirect.com/science/article/pii/0010465571900476 for algorithm description, e.g. alpha,
        betaj and gammaj
        """
        cell_type_flat = self.cell_type.reshape((self.nx * self.ny))

        borders = self._get_borders_array()
        max_i = self.nx * self.ny
        max_j = self.nx * self.ny

        self._ii = []
        self._jj = []
        self._va = []

        alpha = -2 * (1 + (self.dx / self.dy) ** 2)
        print(self.__class__.__name__, ": Creating sparse matrix for field solving.")
        for i in tqdm(xrange(max_i)):  # first only Neumann RB, dirichlet via cell_type (e.g. set borders to phi=0)...
            j = np.floor(i / self.nx) + 1
            if cell_type_flat[i] > 0:  # fixed to phi
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 1:  # buttom
                self._add_for_COO_format(i, i + self.nx, -1.)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 2:  # top
                self._add_for_COO_format(i, i - self.nx, -1.)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 3:  # left
                self._add_for_COO_format(i, i + 1, -1.)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 4:  # right
                self._add_for_COO_format(i, i - 1, -1.)
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

        self.sA = sparse.coo_matrix((self._va, (self._ii, self._jj))).tocsr()

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


    def __init__(self, nz, dz, nr, dr):
        super(CylindricalPoissonSolver, self).__init__()
        print(self.__class__.__name__, ": Created with nx = ", nz, " | dx = ", dz, " | ny = ", nr, " | dy = ", dr)
        self.nz = int(nz)
        self.nr = int(nr)
        self.dz = dz
        self.dr = dr

    def plot_all_fields(self):
        plot_field(self.zvals, self.rvals, self.solved_potential, self.cell_type, 'potential $\phi$ [V]')
        plot_field(self.zvals, self.rvals, self.initial_potential, self.cell_type, 'initial potential $\phi$ [V]')
        plot_field(self.zvals, self.rvals, self.electric_field_r, self.cell_type, 'electric field $E_r$ [V/m]')
        plot_field(self.zvals, self.rvals, self.electric_field_z, self.cell_type, 'electric field $E_z$ [V/m]')
        plot_field(self.zvals, self.rvals, self.electric_field, self.cell_type, 'electric field combinded $E$ [V/m]', lognorm=False)
        for fig in plt.get_fignums():
            plt.figure(fig)
            plt.xlabel('z direction [m]')
            plt.ylabel('r direction [m]')

    def calculate_potential_gauss_seidel(self):
        # set radia
        r = np.zeros_like(np.transpose(self.cell_type))  # get radii in not standard order for gauss-seidel
        for i in range(self.nz):
            for j in range(self.nr):
                r[i][j] = calculate_radius(j, self.dr)

        solved_potential, tol, iters = solve_gauss_seidel_cylindric(np.transpose(self.initial_potential),
                                                                    np.transpose(self.cell_type),
                                                                    np.transpose(self.rho_i), self.dr,
                                                                    self.dz, r, maxit=1e6)
        self.solved_potential = np.transpose(solved_potential)
        print(self.__class__.__name__, ": solve_gauss_seidel_cylindric() reached tolerance of ", tol, " after ", iters, " iterations.")
        return self.solved_potential


    @on_trait_change('solved_potential')
    def calculate_electric_fields(self):
        if self.verbose:
            print(self.__class__.__name__,": Calculating electric fields")
        self.electric_field_z = -np.gradient(self.solved_potential, self.dz, axis=1, edge_order=2)
        self.electric_field_r = -np.gradient(self.solved_potential, self.dr, axis=0, edge_order=2)
        self.electric_field = np.sqrt(self.electric_field_z**2 + self.electric_field_r**2)
        return self.electric_field, self.electric_field_r, self.electric_field_z

    def _create_b_vector(self):
        # creates the right hand side of possion equation
        # rho_i is density of ions in a cell (see poisson equation)
        # See http://www.sciencedirect.com/science/article/pii/0010465571900476 for dx**2 factor (p in paper)
        b = self.initial_potential - self.dz ** 2 * self.rho_i / const.epsilon_0
        b = b.reshape(self.nz * self.nr)
        return b

    @timeitNV
    def create_Ab_matrix(self):
        """
        See https://en.wikipedia.org/wiki/Five-point_stencil
        See http://www.sciencedirect.com/science/article/pii/0010465571900476 for algorithm description, e.g. alpha,
        betaj and gammaj
        """
        cell_type_flat = self.cell_type.reshape((self.nz * self.nr))

        borders = self._get_borders_array()
        max_i = self.nz * self.nr
        max_j = self.nz * self.nr

        self._ii = []
        self._jj = []
        self._va = []

        alpha = -2 * (1 + (self.dz/self.dr)**2)
        print(self.__class__.__name__, ": Creating sparse matrix for field solving.")
        for i in tqdm(xrange(max_i)):  # first only Neumann RB, dirichlet via cell_type (e.g. set borders to phi=0)...
            j = np.floor(i / self.nz)+1
            if cell_type_flat[i] > 0:  # fixed to phi
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 1:  # buttom
                self._add_for_COO_format(i, i + self.nz, -1.)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 2:  # top
                self._add_for_COO_format(i, i - self.nz, -1.)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 3:  # left
                self._add_for_COO_format(i, i + 1, -1.)
                self._add_for_COO_format(i, i, 1.)
            elif borders[i] == 4:  # right
                self._add_for_COO_format(i, i - 1, -1.)
                self._add_for_COO_format(i, i, 1.)
            else:
                betaj = (self.dz / self.dr) ** 2 * (1 + 1 / (2 * j))
                gammaj = (self.dz / self.dr) ** 2 * (1 - 1 / (2 * j))
                if i - 1 >= 0:
                    self._add_for_COO_format(i, i - 1,  1.)
                if i + 1 < max_j:
                    self._add_for_COO_format(i, i + 1, 1.)
                if (i - self.nz) >= 0:  # buttom
                    self._add_for_COO_format(i, i - self.nz, gammaj)
                if (i + self.nz) <= max_j:  # top
                    self._add_for_COO_format(i, i + self.nz, betaj)

                self._add_for_COO_format(i, i, alpha)

        self.sA = sparse.coo_matrix((self._va, (self._ii, self._jj))).tocsr()

    def _electric_field_z_default(self):
        return np.zeros(self._get_mesh_size())

    def _electric_field_r_default(self):
        return np.zeros(self._get_mesh_size())

    def _rvals_default(self):
        return np.linspace(0, self.nr, self.nr, endpoint=False) * self.dr

    def _zvals_default(self):
        return np.linspace(0, self.nz, self.nz, endpoint=False) * self.dz


def cylinder_sugarcube_test(mult=1, nz=340, dz=1e-4, nr=110, dr=1e-4):
    world = CylindricalPoissonSolver(nz=nz * mult, dz=dz / mult, nr=nr * mult, dr=dr / mult)
    world.verbose = True
    # test.do_solver_benchmark()
    ct = world.get_cell_type()

    # r, z
    ct[np.int(60 * mult), 0:np.int(100 * mult + 1.)] = 1  # at r = 60*dx (1e-4), go from z=0, to z=100*dx
    ct[np.int(40 * mult):np.int(60 * mult), np.int(100 * mult)] = 1
    ct[:np.int(60 * mult), 0] = 1

    ct[np.int(70 * mult), 0:np.int(120 * mult + 1)] = 2
    ct[np.int(30 * mult):np.int(70 * mult), np.int(120 * mult)] = 2

    ctd = {'1': 100, '2': 0}  # celltype x = fixed potential
    world.set_cell_type(ct, ctd)
    return world


def cartesian_sugarcube_test(mult=1, nx=340, dx=1e-4, ny=110, dy=1e-4):
    world = CartesianPoissonSolver(nx=nx * mult, dx=dx / mult, ny=ny * mult, dy=dy / mult)
    world.verbose = True
    # test.do_solver_benchmark()
    ct = world.get_cell_type()

    # x, y
    ct[np.int(60 * mult), 0:np.int(100 * mult + 1.)] = 1  # at r = 60*dx (1e-4), go from z=0, to z=100*dx
    ct[np.int(40 * mult):np.int(60 * mult), np.int(100 * mult)] = 1
    ct[:np.int(60 * mult), 0] = 1

    ct[np.int(70 * mult), 0:np.int(120 * mult + 1)] = 2
    ct[np.int(30 * mult):np.int(70 * mult), np.int(120 * mult)] = 2

    ctd = {'1': 100, '2': 0}  # celltype x = fixed potential
    world.set_cell_type(ct, ctd)
    return world


if __name__ == '__main__':
    # world = cylinder_sugarcube_test(mult=1)
    world = cartesian_sugarcube_test(mult=1)
    world.calculate_potential_exact()
    world.plot_all_fields()
    # world.calculate_potential_gauss_seidel()
    # world.plot_all_fields()
    plt.show()