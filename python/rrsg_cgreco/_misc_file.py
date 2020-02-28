# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


## My own functions

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


path = 'python/rrsg_cgreco/rawdata_brain_radial_96proj_12ch.h5'
acc = 1

if not os.path.isfile(path):
    raise ValueError("Data file does not exist")

name = os.path.normpath(path)
with h5py.File(name, 'r') as h5_dataset:
    h5_dataset_rawdata_name = 'rawdata'
    h5_dataset_trajectory_name = 'trajectory'

    if "heart" in name:
        if acc == 2:
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[:, :, :33]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[:, :, :33, :]
        elif acc == 3:
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[:, :, :22]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[:, :, :22, :]
        elif acc == 4:
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[:, :, :11]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[:, :, :11, :]
        else:
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[...]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[...]
    else:
        trajectory = h5_dataset.get(h5_dataset_trajectory_name)[:, :, ::acc]
        rawdata = h5_dataset.get(h5_dataset_rawdata_name)[:, :, ::acc, :]

# Squeeze dummy dimension and transpose to C-style ordering.
rawdata = np.squeeze(rawdata.T)

# Norm Trajectory to the range of (-1/2)/(1/2)
max_trajectory = 2 * np.max(trajectory[0])
trajectory = np.require((trajectory[0] / max_trajectory + 1j * trajectory[1] / max_trajectory).T, requirements='C')

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
fig, ax = plt.subplots()
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
N = None
X = np.real(trajectory[:N])
Y = np.imag(trajectory[:N])
U = np.real(trajectory[:N])
V = np.imag(trajectory[:N])
C = np.angle(trajectory[:N])
Q = ax.quiver(X, Y, U, V, C)

from rrsg_cgreco._helper_fun import goldcomp as goldcomp
from rrsg_cgreco._helper_fun.est_coils import estimate_coil_sensitivities
import rrsg_cgreco.linop as linop
import rrsg_cgreco.solver as solver

"""Extract parameters from data"""
num_coils, num_proj, num_reads = rawdata.shape
# Define parameters...
ogf = 2  # over-gridding factor
num_scans = 1
num_slc = 1
dimX, dimY = (int(num_reads/ogf), int(num_reads/ogf))

DTYPE = np.complex64
DTYPE_real = np.float32

# Density compensation function
dcf = (np.sqrt(np.array(goldcomp.get_golden_angle_dcf(trajectory), dtype=DTYPE_real )).astype(DTYPE_real))
dcf = np.require(np.abs(dcf), DTYPE_real, requirements='C')
plt.imshow(dcf)

par_key = ['num_slc', 'num_scans', 'dimX', 'dimY', 'num_coils', 'N', 'num_proj', 'num_reads', 'dens_cor']
par_val = [num_slc, num_scans, dimX, dimY, num_coils, N, num_proj, num_reads, dcf]
par = dict(zip(par_key, par_val))

# This gives us 'coils' and 'phase_map'
estimate_coil_sensitivities(data=rawdata, trajectory=trajectory, par=par)

plot_3d_list(par['coils'][:,0], augm='np.abs')
plot_3d_list(par['phase_map'], augm='np.angle')
plot_complex_arrows(par['phase_map'][0])

# So now we have some coil sensitivities... awesome.
# How will this be used..? --> Used in the forward operator and adjoint operator (conj) in the operator.
# How will the dcf be used..? --> Used in the gridding process of the NUFFT operator.

# Create some operator
MRImagingOperator = linop.MRIImagingModel(par, trajectory)

# This creates a NUFFT object
NUFFT = linop.NUFFT(par, trajectory, DTYPE=DTYPE, DTYPE_real=DTYPE_real)

# In this NUFFT object we need a regridding kernel. This is based on the KB kernel...
# Based on kwidth, ogf, num reads, klength
plt.plot(NUFFT.kerneltable)
plt.title('kernel table')

# To correct for certain transformation, we want to adopize the data as welll
# Based on kerneltable FT
plt.imshow(NUFFT.deapo)
plt.title('deapodiztion')

# Adjoint NUFFT operation
# Grid k-space
ogkspace = NUFFT._grid_lut(trajectory[np.newaxis])

# FFT
ogkspace = np.fft.ifftshift(ogkspace, axes=NUFFT.fft_dim)
ogkspace = np.fft.ifft2(ogkspace, norm='ortho')
ogkspace = np.fft.ifftshift(ogkspace, axes=NUFFT.fft_dim)
result_adjoint = NUFFT._deapo_adj(ogkspace)

## Adjoint of operator MRI
np.sum(MRImagingOperator.NUFFT.adjoint(inp) * MRImagingOperator.conj_coils, 1)

# forward NUFFT operation
ogkspace = NUFFT._deapo_fwd(inp)
# FFT
ogkspace = np.fft.fftshift(ogkspace, axes=(-2, -1))
ogkspace = np.fft.fft2(ogkspace, norm='ortho')
ogkspace = np.fft.fftshift(ogkspace, axes=(-2, -1))
# Resample on Spoke
NUFFT._invgrid_lut(ogkspace)
## Forward operator of MRI
MRImagingOperator.NUFFT.forward(inp * MRImagingOperator.coils)


# When are forward and adjoint being used..?

# Well lets find out in the optimizqation thingy!

cgs = solver.CGReco(par)
cgs.set_operator(MRImagingOperator)

# Start reconstruction
plt.imshow(np.real(rawdata * par["dens_cor"])[0])
ogkspace = NUFFT._grid_lut(rawdata[None, :, None])
recon_result = cgs.optimize((rawdata * par["dens_cor"]))

b = MRImagingOperator.adjoint(inp)
Ax = MRImagingOperator.adjoint(MRImagingOperator.forward(b[None, ...]))


# Next step is to create an operator to solve our problems...
# This object MRI... stores three new objects
# - NUFFT operator (forward/adjoint)
        # This guy on its turn gets
            # Keiser Bessel Kernel
            # Deapodization stuff
            # Implementas forward/adjoint operators
                # The forward/adjoint steps contain apodization and regridding. Show how this works.

#   coil sensitivities
#   conjugate of these coil sensitivities

# But it also implements adjoint.forward operators..
    # This is based on the forward/adjoint operators of the NUFFT object.


# Besides the operator, we also need a way to solve our equaition Ax=b.
# This is done by defining an CG recon object
# This is just a simple and generic CG approach

# Tada and done..
# Result..?
# Result without regreidding/adopization?

# TODO make a print dict function to make clear what is in this par dictionary
# TODO make clear what dimensions are used where. Somewhere (for rawdata) at certain locations new axes are added. It is not clear directly why this is.
