from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np

from traits.api import HasTraits, on_trait_change, Float, Array

from ElectricFieldSolver import CylindricalPoissonSolver


if __name__ == '__main__':
    # read bitmap
    img = np.transpose(plt.imread('examples/DepressedCollector.bmp'))
    plt.figure()
    plt.imshow(img, origin='lower')
    plt.colorbar()
    plt.show()

    dz = 1.5/img.shape[1]  # 1m
    dr = 1./img.shape[0]  # 1m

    world = CylindricalPoissonSolver(img.shape[1], dz, img.shape[0], dr)

    ctd = {'187': 250e3, '213': 170e3, '113': 20e3, '68':-50e3}  # celltype x = fixed potential
    world.set_cell_type(img, ctd)
    world.calculate_potential_exact()
    world.plot_all_fields()
    plt.show()
