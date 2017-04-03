from __future__ import division, print_function
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from traits.api import HasTraits, on_trait_change, Float, Array

from ElectricFieldSolver import CylindricalPoissonSolver, CartesianPoissonSolver

def depressed_collector_example():
    # read bitmap
    img = np.transpose(plt.imread('examples/DepressedCollector/DepressedCollector.bmp'))

    # have a look at color values (hover with mouse)..
    plt.figure()
    plt.imshow(img, origin='lower')
    plt.colorbar()
    plt.show()

    dz = 1.5/img.shape[1]  # 1m
    dr = 1./img.shape[0]  # 1m

    world = CylindricalPoissonSolver(img.shape[1], dz, img.shape[0], dr)

    # assign values a voltage [V]
    ctd = {'187': 250e3, '213': 170e3, '113': 20e3, '68':-50e3}  # celltype x = fixed potential
    world.set_electric_cell_type(img, ctd)
    world.calculate_potential_exact()
    savedict = world.plot_all_fields()
    plt.show()


def electron_gun_example():
    # read bitmap
    img = np.transpose(plt.imread('examples/ElectronGun/ElectronGun.bmp'))

    # have a look at color values (hover with mouse)..
    plt.figure()
    plt.imshow(img, origin='lower')
    plt.colorbar()
    # plt.show()

    dz = (0.205 + 0.05)/img.shape[1]
    dr = 0.1425/img.shape[0]


    world = CylindricalPoissonSolver(img.shape[1], dz, img.shape[0], dr)

    # split for currents and potentials (or have two images)
    img_potential = np.zeros_like(world.get_electric_cell_type())
    img_magnetic = np.zeros_like(world.get_magnetic_cell_type())

    img_magnetic = np.where(img <= 69, img, img_magnetic)
    ct_magn = {'40': 2e7, '54': 2e7, '69': 2e7}

    img_potential = np.where(img > 69, img, img_potential)
    ct_pot = {'112': -7000., '161': 0.}


    world.set_electric_cell_type(img_potential, ct_pot)
    world.set_magnetic_cell_type(img_magnetic, ct_magn)
    world.calculate_potential_exact()
    savedict = world.plot_all_fields()
    plt.show()


def electrode_example():
    # img = plt.imread('examples/ElectrodeCharge/Electrode_Configuration.bmp')
    # img = plt.imread('examples/ElectrodeCharge/Electrode_Configuration - bigger.bmp')
    # img = plt.imread('examples/ElectrodeCharge/Electrode_Configuration - bigger 2.bmp')
    img = np.transpose(np.rot90(plt.imread('examples/ElectrodeCharge/Electrode_Configuration - try 2.bmp'),3))

    plt.figure()
    plt.imshow(img, origin='lower')
    plt.colorbar()

    plt.show()

    dx = 300e-6/img.shape[1]
    dy = 300e-6/img.shape[0]

    world = CartesianPoissonSolver(img.shape[1], dx, img.shape[0], dy)

    # assign values a voltage [V]
    ctd1 = {'255': -0.5, '177': -3.5, '100': 0., '75': 2, '161':0, '153':-21}  # celltype x = fixed potential
    ctd2 = {'255': 0, '177': -2.5, '100': 2, '75': 2}  # celltype x = fixed potential
    world.set_electric_cell_type(img, ctd1)
    world.calculate_potential_exact()
    savedict = world.plot_all_fields(figsize=(10,8))

    for k in savedict:
        np.savetxt('examples/ElectrodeCharge/results/' + k + '_config_new.txt', savedict[k])

    for fignum in plt.get_fignums():
        plt.figure(fignum)
        plt.savefig('examples/ElectrodeCharge/results/' + savedict.keys()[fignum] + '_config_new.png', dpi=300)

    plt.show()


def philips_source():
    img = np.rot90(plt.imread('examples/Philipssource/Electrode_cropped_simplified_ready.bmp'), 3)

    plt.figure()
    plt.imshow(img, origin='lower')
    plt.colorbar()

    dx = 10e-3/112
    dy = dx

    world = CartesianPoissonSolver(img.shape[1], dx, img.shape[0], dy)
    ctd = {'102': 0, '127': -2500, '153': -3000, '69': -2500, '229':0}
    world.set_electric_cell_type(img, ctd)
    world.calculate_potential_exact()
    savedict = world.plot_all_fields(figsize=(10, 8))
    plt.show()


def sugar_cube_cylinder():
    img = plt.imread('examples/SugarCube/sugarcube_example.bmp')
    print("image shape = ", img.shape)
    plt.figure()
    plt.imshow(img, origin='lower', interpolation=None)
    plt.colorbar()

    dz = 1e-3
    dr = 1e-3

    world = CylindricalPoissonSolver(img.shape[1], dz, img.shape[0], dr)

    # assign values a voltage [V]
    ctd = {'153': 0, '178': 100}  # celltype x = fixed potential
    world.set_electric_cell_type(img, ctd)
    world.calculate_potential_exact()
    savedict = world.plot_all_fields()
    plt.show()


def sugar_cube_cartesian():
    img = plt.imread('examples/SugarCube/sugarcube_example_cartesian.bmp')
    print("image shape = ", img.shape)
    plt.figure()
    plt.imshow(img, origin='lower', interpolation=None)
    plt.colorbar()

    dx = 1e-3
    dy = 1e-3

    world = CartesianPoissonSolver(img.shape[1], dx, img.shape[0], dy)

    # assign values a voltage [V]
    ctd = {'153': 0, '178': 100}  # celltype x = fixed potential
    world.set_electric_cell_type(img, ctd)
    world.calculate_potential_exact()
    savedict = world.plot_all_fields()

    for fignum in plt.get_fignums():
        plt.figure(fignum)
        current_limits = plt.ylim()
        # print(current_limits)
        plt.ylim(current_limits[1]/2.,current_limits[1])


    plt.show()


if __name__ == '__main__':
    # depressed_collector_example()
    # electron_gun_example()
    electrode_example()
    # philips_source()
    # sugar_cube_cylinder()
    # sugar_cube_cartesian()