from __future__ import print_function, division
import numpy as np
import numba as nb
import re, os, time
import matplotlib.pyplot as plt


@nb.jit(nopython=True, nogil=True)
def calculate_radius(i, dr):
    return i*dr


def get_numpy_circle(dinpix=200, center=100, thickness_factor=10):
    xx, yy = np.mgrid[:dinpix, :dinpix]
    circle = (xx - center) ** 2 + (yy - center) ** 2
    d = (xx.shape[0] ** 2 + xx.shape[1] ** 2)/ 10.
    donut = np.logical_and(circle < (d + d/thickness_factor), circle > (d - d/thickness_factor))
    return donut



@nb.jit(nopython=True, nogil=True)
def jit_cross(a, b):
    # cross product which is faster than numpy.cross (about 4 times for different vector sizes)
    c = np.zeros(b.shape, dtype=a.dtype)
    for i in xrange(b.shape[0]):
        c[i,0] = a[i,1]*b[i,2] - a[i,2]*b[i,1]
        c[i,1] = a[i,2]*b[i,0] - a[i,0]*b[i,2]
        c[i,2] = a[i,0]*b[i,1] - a[i,1]*b[i,0]

    return c

def plot_field(xvals, yvals, data, cell_type, zlabel, grid=False, lognorm=False):
    import copy
    data = copy.deepcopy(data)
    if np.any(data) > 0:
        data[np.where(cell_type > 0)] = np.nan

    fig1 = plt.figure(figsize=(12, 4), facecolor='w', edgecolor='k', tight_layout=True)
    n_levels = 100

    # plt.imshow(data, origin='lower')
    if lognorm:
        from matplotlib import ticker
        from matplotlib.colors import LogNorm
        # cf = plt.contourf(self.yvals, self.xvals, data, n_levels, alpha=.75, linewidth=1, cmap='jet', locator=ticker.LogLocator(numticks=n_levels))
        plt.imshow(data, norm=LogNorm(), origin='lower')
        plt.colorbar(label=zlabel, format='%.2e')
    else:
        # print(data)
        cf = plt.contourf(xvals, yvals, data, n_levels, alpha=.75, linewidth=1, cmap='jet')
        # cf.set_clim(np.nanmin(data), np.nanmax(data))
        plt.colorbar(label=zlabel, format='%.2e')

        (RR, ZZ) = np.meshgrid(xvals, yvals)  # r, z


        celltype_gr_0 = np.where(cell_type > 0.)  # find all cell types which are not 0
        # plt.scatter(RR[celltype_gr_0], ZZ[celltype_gr_0],
        #             c=cell_type[celltype_gr_0], s=50, cmap='bwr_r')

        plt.ylim(min(yvals), max(yvals))
        plt.xlim(min(xvals), max(xvals))

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if grid: plt.grid(b=True, which='both', color='k', linestyle='-')


def find_unique_rows_in_array(a):
    # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    # completly autistic...
    return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])


def get_indices_of_vector(matrix, vec):
    # finds a vector in a matrix (array of vectors)
    matrix = np.ascontiguousarray(matrix)
    dt = np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[-1]))
    e_view = matrix.view(dt)
    search = np.array(vec, dtype=matrix.dtype).view(dt)
    mask = np.in1d(e_view, search)
    indices = np.unravel_index(np.where(mask), matrix.shape[:-1])
    return indices


def timeitNV(f):
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        diff = te-ts
        unit = 'sec'
        if diff < 1:
            diff *= 1000.
            unit = ' msec'

        print('func:%r took: %2.4f %s' %(f.__name__, diff, unit))
        return result

    return timed


