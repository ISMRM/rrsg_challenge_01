# encoding: utf-8

import math
import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Collection of functions to make the plotting of high dimensional data easier.
"""


def simple_div(n):
    """
    Calculate the divisors of n. Used to split arrays into a groups of integer size, or to asses how
    large the batch size should be when training a model
    :param n:
    :return:
    """
    return [i for i in range(n, 0, -1) if n % i == 0]


def get_square(x):
    """
    Used to get an approximation of the square of a number.
    Needed to place N plots on a equal sized grid.
    :param x:
    :return:
    """
    x_div = simple_div(x)
    x_sqrt = math.sqrt(x)
    diff_list = [abs(y - x_sqrt) for y in x_div]
    res = diff_list.index(min(diff_list))
    return x_div[res], x // x_div[res]


def plot_sequence(image_list, **kwargs):
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

    debug = kwargs.get('debug', False)

    f = plt.figure(fignum, figsize, dpi)
    f.suptitle(title_string)

    n_rows = len(image_list)
    gs0 = gridspec.GridSpec(n_rows, 1, figure=f)
    gs0.update(wspace=0.1, hspace=0.1)  # set the spacing between axes.

    print('amount of rows..', n_rows)
    for i, i_gs in enumerate(gs0):
        temp_img = image_list[i]

        if hasattr(temp_img, 'ndim') and hasattr(temp_img, 'shape') and hasattr(temp_img, 'reshape'):
            if temp_img.ndim == 4:
                n_sub_col = temp_img.shape[0]
                n_sub_row = temp_img.shape[1]
                temp_img = temp_img.reshape((n_sub_col * n_sub_row, ) + temp_img.shape[2:])
            elif temp_img.ndim == 3:
                n_sub_col = temp_img.shape[0]
                if n_sub_col > 8:
                    n_sub_col, n_sub_row = get_square(n_sub_col)
                else:
                    n_sub_row = 1
            else:
                temp_img = temp_img[np.newaxis]
                n_sub_col = 1
                n_sub_row = 1
        else:
            n_sub_col = len(temp_img)
            n_sub_row = 1

        for j, ii_gs in enumerate(i_gs.subgridspec(n_sub_row, n_sub_col)):
            ax = f.add_subplot(ii_gs)
            if augm_ind:
                plot_img = eval('{fun}({var})'.format(fun=augm_ind, var=str('temp_img[j]')))
            else:
                plot_img = temp_img[j]

            if debug:
                print(f'shape {i} - {temp_img.shape}', end=' \t|\t')
                print(f'row/col {n_sub_row} - {n_sub_col}', end=' \t|\t')
                print(f'shape {j} - {plot_img.shape}', end=' \t|\n')

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


def plot_complex_arrows(x, xmin=(-0.5, 0.5), ymin=(-0.5, 0.5)):
    fig, ax = plt.subplots()
    ax.set_xlim(xmin)
    ax.set_ylim(ymin)
    U = X = np.real(x)
    V = Y = np.imag(x)
    C = np.angle(x)
    ax.quiver(X, Y, U, V, C)
    return fig, ax