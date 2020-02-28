# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rrsg_cgreco.recon
from rrsg_cgreco._helper_fun import goldcomp as goldcomp
from rrsg_cgreco._helper_fun.est_coils import estimate_coil_sensitivities
from rrsg_cgreco._helper_fun.calckbkernel import calculate_keiser_bessel_kernel
import rrsg_cgreco.linop as linop
import rrsg_cgreco.solver as solver


""" Own funcitons"""

def plot_3d_list(image_list, **kwargs):
    # Input of either a 2d list of np.arrays.. or a 3d list of np.arrays..

    figsize = kwargs.get('figsize')
    fignum = kwargs.get('fignum')
    dpi = kwargs.get('dpi')

    title_string = kwargs.get('title', "")
    sub_title = kwargs.get('subtitle', None)
    cbar_ind = kwargs.get('cbar', False)

    vmin = kwargs.get('vmin', None)
    ax_off = kwargs.get('ax_off', False)
    augm_ind = kwargs.get('augm', None)
    aspect_mode = kwargs.get('aspect', 'equal')

    f = plt.figure(fignum, figsize, dpi)
    f.suptitle(title_string)

    n_rows = len(image_list)
    gs0 = gridspec.GridSpec(n_rows, 1, figure=f)
    gs0.update(wspace=0.1, hspace=0.1)  # set the spacing between axes.

    for i, i_gs in enumerate(gs0):
        temp_img = image_list[i]
        if isinstance(temp_img, np.ndarray):
            if temp_img.ndim == 3:
                n_col = temp_img.shape[0]
            else:
                temp_img = temp_img[np.newaxis]
                n_col = 1
        else:
            n_col = len(temp_img)

        for j, ii_gs in enumerate(i_gs.subgridspec(1, n_col)):
            ax = f.add_subplot(ii_gs)
            if augm_ind:
                plot_img = eval('{fun}({var})'.format(fun=augm_ind, var=str('temp_img[j]')))
            else:
                plot_img = temp_img[j]

            map_ax = ax.imshow(plot_img, vmin=vmin, aspect=aspect_mode)

            if cbar_ind:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                temp_cbar = plt.colorbar(map_ax, cax=cax)
                if vmin is None:
                    vmin_temp = [np.min(plot_img), np.max(plot_img)]
                    map_ax.set_clim(vmin_temp)
                    temp_cbar.set_ticks(vmin_temp)
                else:
                    map_ax.set_clim(vmin)
                    temp_cbar.set_ticks(vmin)

                # vmin = None  # otherwise they all get the same thing
            if sub_title is not None:
                ax.set_title(sub_title[i][j])
            if ax_off:
                ax.set_axis_off()

    return f


def plot_complex_arrows(x):
    fig, ax = plt.subplots()
    U = X = np.real(x)
    V = Y = np.imag(x)
    C = np.angle(x)
    ax.quiver(X, Y, U, V, C)
    return Q


""" Read in the data """

path = 'python/rrsg_cgreco/rawdata_brain_radial_96proj_12ch.h5'
acceleration = 1

# This gives us normalized trajectory over range -0.5 .. 0.5
rawdata, trajectory = rrsg_cgreco.recon.read_data(path=path, acc=acceleration)


""" Display trajectory """

trajectory = np.squeeze(trajectory)
# Example of a couple of spokes
plt.plot(np.real(trajectory).T, 'b', alpha=0.5)
plt.plot(np.imag(trajectory).T, 'r', alpha=0.5)

# Imshow of trajectory.. Not that usefull
fig, ax = plt.subplots(1, 2, figsize=(20, 20))
ax[0].imshow(np.real(trajectory))
ax[0].set_title('real part')
ax[1].imshow(np.imag(trajectory))
ax[1].set_title('imaginary part')

# Quiver plot of the trajectory. This is insightful
fig_quiver, ax_quiver = plt.subplots()
ax_quiver.set_xlim(-.5, .5)
ax_quiver.set_ylim(-.5, .5)
N = None
U = X = np.real(trajectory[:N])
V = Y = np.imag(trajectory[:N])
C = np.angle(trajectory[:N])
Q = ax_quiver.quiver(X, Y, U, V, C)


""" Display raw data """

plot_3d_list(rawdata, augm='np.abs')


""" NUFFT step by step guide """

""" We are going to solve an inverse problem b = Ax
    We need several objects for that
    
    - density compensation function
    - Coil sensitivities, derived from measurement..
    - adopization function
    - regridding procedure
    - adjoint/forward operator creation
    - optimization/solver algorithm  """


""" We start with defining parameters for this process"""

num_coils, num_proj, num_reads = rawdata.shape
overgridfactor = 2  # over-gridding factor
num_scans = 1
num_slc = 1
dimX, dimY = (int(num_reads/overgridfactor),
              int(num_reads/overgridfactor))  # Hoe en waar worden dimX en dimY gebruikt?

DTYPE = np.complex64
DTYPE_real = np.float32

# Put these data points in a dictionary
par_key = ['num_slc', 'num_scans', 'dimX', 'dimY', 'num_coils', 'num_proj', 'num_reads', 'overgridfactor']
par_val = [num_slc, num_scans, dimX, dimY, num_coils, num_proj, num_reads, overgridfactor]
par = dict(zip(par_key, par_val))

""" Get the density compensation function based on the trajectory data """

density_comp = (np.sqrt(np.array(goldcomp.get_golden_angle_dcf(trajectory), dtype=DTYPE_real)).astype(DTYPE_real))
density_comp = np.require(np.abs(density_comp), DTYPE_real, requirements='C')
plt.imshow(density_comp)

par['dens_cor'] = density_comp

""" Get Bessel Kernel for interpolation """

par['kwidth'] = 5
par['klength'] = 500
kerneltable, kerneltable_FT, u = calculate_keiser_bessel_kernel(G=256, **par)
plt.plot(u, kerneltable)


""" Get adopization """

deapodization = 1 / kerneltable_FT.astype(DTYPE_real)
deapodization = np.outer(deapodization, deapodization)
plt.imshow(deapodization)


""" However, we can also combine all this information into one module! """

# This creates a NUFFT object
import importlib
importlib.reload(linop)
NUFFT = linop.NUFFT(par, trajectory[np.newaxis], DTYPE=DTYPE, DTYPE_real=DTYPE_real)

ogkspace, grid_point_mapping = NUFFT._grid_lut(rawdata[None, :, None], return_mapping=True)

""" What did this gridding exactly do? Well it created a mapping! """

# This can take some time.. depending on n_step
n_step = 30
for jj in range(0, 92, n_step):
    for i_point in range(jj * 512, (jj+1)*512, 32):
        for i, temp_point in enumerate(grid_point_mapping[i_point]):
            if i == 0:
                # Original point
                plt.scatter(temp_point[0], temp_point[1], c='r')
            else:
                # Mapped point
                plt.scatter(temp_point[0], temp_point[1], c='k')

""" And what is the result of the gridding to the original data...? """

plot_3d_list(rawdata, augm='np.real', vmin=(0, 0.000001))
plot_3d_list(ogkspace[0, :, 0], augm='np.real', vmin=(0, 0.000001))

""" Now let us add a Fourier Transform to it"""
res = NUFFT.adjoint(rawdata[None, :, None])  # Somehwere it should be made clear how these dimensions are defined.
plot_3d_list(res[0, :, 0], augm='np.angle')
plot_3d_list(res[0, :, 0], augm='np.abs')

# Could show with and without apodization
# Amazing! We moved stuff to a cartesian grid and got ourselves an image!

""" Move on to MRI Imaging Model """

""" Compute the coil sensitivities """
# This adds the 'coils' and 'phase_map' keys to the par dict.
estimate_coil_sensitivities(data=rawdata, trajectory=trajectory, par=par)

# I dont understand yet why this is 250 x 250 now..
plot_3d_list(par['coils'][:, 0], augm='np.abs')
plot_3d_list(par['phase_map'], augm='np.angle')

""" The adjoint operator depends on the implementation of NUFFT... """

MRImagingOperator = linop.MRIImagingModel(par, trajectory)

# Adjoint of operator MRI
#
res = np.sum(MRImagingOperator.NUFFT.adjoint(rawdata[None, :, None]) * MRImagingOperator.conj_coils, 1)

""" Gradually move towards solving with CG """

# TODO make a print dict function to make clear what is in this par dictionary
# TODO make clear what dimensions are used where. Somewhere (for rawdata) at certain locations new axes are added. It is not clear directly why this is.
