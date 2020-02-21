#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:28:16 2019

@author: omaier

Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this par["file"] except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np
import sys
from python.rrsg_cgreco._helper_fun import nlinvns
from python.rrsg_cgreco._helper_fun import goldcomp
from python.rrsg_cgreco import linop

# Estimates sensitivities and complex image.
# (see Martin Uecker: Image reconstruction by regularized nonlinear
# inversion joint estimation of coil sensitivities and image content)
DTYPE = np.complex64
DTYPE_real = np.float32


def estimate_coil_sensitivities(data, trajectory, par):
    """
      Estimate complex coil sensitivities using NLINV from Martin Uecker et al.

      Estimate complex coil sensitivities using NLINV from Martin Uecker et al.
      Non-uniform data is first regridded and subsequently transformed back to
      k-space using a standard fft.
      The result ist stored in the parameter (par) dict. Internally the nlinvns
      function is called.

      This is just a workaround for now to allow
      for fast coil estimation. The final script will use precomputed
      profiles most likely from an Espirit reconstruction.


    Args
    ----
        data (numpy.array):
          complex k-space data
        trajectory (numpy.array):
          complex trajectory information
        par:
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (num_slc),
            number of scans (num_scans), image dimensions (dimX, dimY), number
            of coils (num_coils), sampling pos (N) and read outs (NProj).

    """
    nlinvNewtonSteps = 6
    nlinvRealConstr = False

    new_shape = (par["num_scans"] * par["num_proj"], par["num_reads"])
    traj_coil = np.reshape(trajectory, new_shape)
    dens_cor_coil = np.sqrt(np.array(goldcomp.cmp(traj_coil), dtype=DTYPE))

    C_shape = (par["num_coils"], par["num_slc"], par["dimY"], par["dimX"])
    par["C"] = np.zeros(C_shape, dtype=DTYPE)
    phase_map_shape = (par["num_slc"], par["dimY"], par["dimX"])
    par["phase_map"] = np.zeros(phase_map_shape, dtype=DTYPE)

    par_coils = {}
    par_coils["dens_cor"] = dens_cor_coil
    par_coils["num_reads"] = par["num_reads"]
    par_coils["num_scans"] = 1
    par_coils["num_coils"] = par["num_coils"]
    par_coils["num_slc"] = 1
    par_coils["dimX"] = par["dimX"]
    par_coils["dimY"] = par["dimY"]
    par_coils["num_proj"] = par["num_proj"]
    FFT = linop.NUFFT(par_coils, traj_coil[None, ...])

    combined_data = np.transpose(
        data[None, :, None, ...], (1, 0, 2, 3, 4))
    combined_data = np.reshape(
                      combined_data,
                      (1,
                       par["num_coils"],
                       1,
                       par["num_scans"] * par["num_proj"],
                       par["num_reads"])) * dens_cor_coil
    combined_data = FFT.adj(combined_data)
    combined_data = np.fft.fft2(combined_data,
                                norm='ortho')

    for i in range(0, (par["num_slc"])):
        sys.stdout.write(
            "Computing coil sensitivity map of slice %i \r" %
            (i))
        sys.stdout.flush()

        result = nlinvns.nlinvns(
                    np.squeeze(combined_data[:, :, i, ...]),
                    nlinvNewtonSteps,
                    True,
                    nlinvRealConstr)

        par["coils"][:, i, :, :] = result[2:, -1, :, :]
        sys.stdout.write("slice %i done \r"
                         % (i))
        sys.stdout.flush()
        if not nlinvRealConstr:
            par["phase_map"][i, :, :] = np.exp(
                1j * np.angle(result[0, -1, :, :]))

            # standardize coil sensitivity propar["file"]s
    sumSqrC = np.sqrt(
        np.sum(
            (par["coils"] *
             np.conj(
                par["coils"])),
            0))  # 4, 9, 128, 128
    par["in_scale"] = sumSqrC
    if par["num_coils"] == 1:
        par["coils"] = sumSqrC
    else:
        par["coils"] = par["coils"] / \
          np.tile(sumSqrC, (par["num_coils"], 1, 1, 1))
