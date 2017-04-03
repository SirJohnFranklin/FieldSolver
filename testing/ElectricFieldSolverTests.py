from __future__ import print_function, division

from ElectricFieldSolver import CylindricalPoissonSolver, CartesianPoissonSolver
from HelperFunctions import plot_field

import matplotlib.pyplot as plt
import numpy as np


def benchmark_matrix_direct_solve():
    """ 2017-04-03
    
        benchmark_matrix_direct_solve: Testing dx =  0.002  | dy =  0.003
        CylindricalPoissonSolver : Created with nx =  340  | dx =  0.002  | ny =  110  | dy =  0.003
        
        -m cProfile -s tottime
        ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        solvers with sparse matrix preparation
        1    0.000    0.000    1.343    1.343 ElectricFieldSolver.py:175(calculate_potential_exact)
                2    0.435    0.218    0.791    0.396 ElectricFieldSolver.py:585(create_Ab_matrix)
                2    0.326    0.163    0.326    0.163 {scipy.sparse.linalg.dsolve._superlu.gssv}
        1  800.569  800.569  800.569  800.569 ElectricFieldSolver.py:19(solve_gauss_seidel_cylindric)

    """
    for dz in [2e-3, 5e-3]:
        for dr in [3e-3, 4e-3, 5e-3]:
            print("benchmark_matrix_direct_solve: Testing dx = ", dz, " | dy = ", dr)
            world = cylinder_sugarcube_test(mult=1, dz=dz, dr=dr)
            ergspsolve = world.calculate_potential_exact()
            ergseidel = world.calculate_potential_gauss_seidel()
            if not np.allclose(ergspsolve, ergseidel, rtol=1e-6):  # atol is set higher, since gauss-seidel only checks rtol
                print("rtol is higher than 1e-6")
                plot_field(world.zvals, world.rvals, ergspsolve, world.cell_type, 'potential $\phi$ [V] (spsolve)')
                plot_field(world.zvals, world.rvals, ergseidel, world.cell_type, 'potential $\phi$ [V] (gauss-seidel)')
                plt.figure()
                plt.imshow((ergspsolve-ergseidel)/(ergspsolve+ergseidel), origin='lower', interpolation="nearest")
                plt.colorbar(format='%.0e')
                plt.show()

    # 2016-10-18: passed.


def cylinder_sugarcube_test(mult=1, nz=340, dz=1e-4, nr=110, dr=1e-4, currents=True):
    world = CylindricalPoissonSolver(nz=nz * mult, dz=dz / mult, nr=nr * mult, dr=dr / mult)
    world.verbose = True
    # test.do_solver_benchmark()
    ct = world.get_electric_cell_type()

    # r, z
    ct[np.int(60 * mult), 0:np.int(100 * mult + 1.)] = 1  # at r = 60*dx (1e-4), go from z=0, to z=100*dx
    ct[np.int(40 * mult):np.int(60 * mult), np.int(100 * mult)] = 1
    ct[:np.int(60 * mult), 0] = 1

    ct[np.int(70 * mult), 0:np.int(120 * mult + 1)] = 2
    ct[np.int(30 * mult):np.int(70 * mult), np.int(120 * mult)] = 2

    ctd = {'1': 100, '2': 0}  # celltype x = fixed potential
    world.set_electric_cell_type(ct, ctd)

    if currents:
        cc = world.get_magnetic_cell_type()
        cc[40:50, 160:240] = 1
        ccd = {'1': 1./(dr * dz)}
        world.set_magnetic_cell_type(cc, ccd)

    return world


def cartesian_sugarcube_test(mult=1, nx=340, dx=1e-4, ny=110, dy=1e-4, currents=True):
    world = CartesianPoissonSolver(nx=nx * mult, dx=dx / mult, ny=ny * mult, dy=dy / mult)
    world.verbose = True
    # test.do_solver_benchmark()
    ct = world.get_electric_cell_type()

    # x, y
    ct[np.int(60 * mult), 0:np.int(100 * mult + 1.)] = 1  # at r = 60*dx (1e-4), go from z=0, to z=100*dx
    ct[np.int(40 * mult):np.int(60 * mult), np.int(100 * mult)] = 1
    ct[:np.int(60 * mult), 0] = 1

    ct[np.int(70 * mult), 0:np.int(120 * mult + 1)] = 2
    ct[np.int(30 * mult):np.int(70 * mult), np.int(120 * mult)] = 2

    ctd = {'1': 100, '2': 0}  # celltype x = fixed potential
    world.set_electric_cell_type(ct, ctd)

    if currents:
        cc = world.get_magnetic_cell_type()
        cc[40:50, 160:240] = 1
        ccd = {'1': 1/(dx * dy)}
        world.set_magnetic_cell_type(cc, ccd)

    return world


def test_magnetic_field_solver(multr=1, multz=1, nz=100, dz=1e-3, nr=120, dr=1e-3):
    world = CylindricalPoissonSolver(nz=nz * multz, dz=dz / multz, nr=nr * multr, dr=dr / multr)

    cc = world.get_magnetic_cell_type()
    cc[45*multr:55*multr, 20*multz:80*multz] = 1
    ccd = {'1': 2e7}
    world.set_magnetic_cell_type(cc, ccd)
    return world


if __name__ == '__main__':
    benchmark_matrix_direct_solve()


    # world = cylinder_sugarcube_test(mult=8, currents=True)
    # ergspsolve = world.calculate_potential_exact()
    # ergseidel = world.calculate_potential_gauss_seidel()
    # world.plot_all_fields()

    # ergspsolve = world.calculate_potential_exact()
    # world.plot_all_fields()
    # print()
    # print(np.allclose(ergspsolve, ergseidel, atol=1e-3))

    # world = test_magnetic_field_solver(multr=7, multz=4)
    # world.calculate_potential_exact()
    # world.plot_all_fields()

    # world = cylinder_sugarcube_test(mult=1, currents=True)
    # world = cartesian_sugarcube_test(mult=1)

    plt.show()



