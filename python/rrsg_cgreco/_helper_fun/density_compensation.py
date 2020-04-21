#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:05:59 2018

@author: omaier

Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import numpy as np


def get_density_from_gridding(data_par, gridding_matrix):
    """
    Density compensation based on the used trajectory.

    Args
    ----
      data_par (dict):
         Dictionary holding data propierties
      gridding_matrix (sparse_matrix):
          Sparse matrix realizing the gridding via matrix-vector 
          multiplication.

    Returns
    -------
      numpy.array
        Density compensation
    """
    density = gridding_matrix.transpose()@(
        np.ones(gridding_matrix.shape[0]))
    density /= np.max(density)
    density[density!=0] = 1/density[density!=0]
    density = gridding_matrix@density
    density = np.reshape(
        density, 
        (data_par["num_proj"], data_par["num_reads"])
        )
    return density.astype(data_par["DTYPE_real"])


def get_golden_angle_dcf(k):
    """
    Golden angle density compensation.

    Args
    ----
      k (numpy.array):
         Complex k-space trajectory

    Returns
    -------
      numpy.array
        Ramp for golden angle density compensation
    """
    if len(np.shape(k)[:-1]) == 2:
        nspokes, N = np.shape(k)[:-1]
    elif len(np.shape(k)[:-1]) == 3:
        NScan, nspokes, N = np.shape(k)[:-1]
    else:
        raise ValueError("Passed trajectory has the wrong "
                         "number of dumensions.")

    w = np.abs(np.linspace(-N/2, N/2, N))/(N/2)  # ramp from -1...1
    w = np.repeat(w, nspokes, 0)
    # w /= np.min(w)
    w *= (N * np.pi / 4) / nspokes
    w = np.reshape(w, (N, nspokes)).T
    return w


def get_voronoi_dcf(trajectory):
    """
    Made something. Needs more checks and everything. Oh and a correction for the outer cells.
    That scales weirdly with the other cells
    Args:
        trajectory:

    Returns:

    """
    import scipy.spatial
    temp = np.array([trajectory.real, trajectory.imag]).reshape((-1, 2))
    res = scipy.spatial.Voronoi(temp.T)
    scipy.spatial.voronoi_plot_2d(res)

    n_points = len(res.points)
    dcf = np.zeros(n_points)
    for i_point in range(n_points):
        i_region = res.point_region[i_point]
        indices = res.regions[i_region]
        dcf[i_point] = scipy.spatial.ConvexHull(res.vertices[indices]).volume

    dcf_reshp = dcf.reshape(trajectory.shape)
    return dcf_reshp