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

    Args:
        data:
        trajectory:
        par:

    Returns:

    """
    nlinvNewtonSteps = 6
    nlinvRealConstr = False

    traj_coil = np.reshape(
        trajectory, (par["NScan"] * par["Nproj"], par["nFE"]))
    dcf_coil = np.sqrt(
        np.array(
            goldcomp.cmp(traj_coil),
            dtype=DTYPE))

    par["C"] = np.zeros(
        (par["NC"], par["NSlice"], par["dimY"], par["dimX"]),
        dtype=DTYPE)
    par["phase_map"] = np.zeros(
        (par["NSlice"], par["dimY"], par["dimX"]), dtype=DTYPE)

    par_coils = {}
    par_coils["traj"] = traj_coil[None, ...]
    par_coils["dcf"] = dcf_coil
    par_coils["nFE"] = par["nFE"]
    par_coils["NScan"] = 1
    par_coils["NC"] = par["NC"]
    par_coils["NSlice"] = 1
    par_coils["dimX"] = par["dimX"]
    par_coils["dimY"] = par["dimY"]
    par_coils["Nproj"] = par["Nproj"]
    FFT = linop.NUFFT(par_coils)

    combinedData = np.transpose(
        data[None, :, None, ...], (1, 0, 2, 3, 4))
    combinedData = np.reshape(
                      combinedData,
                      (1,
                       par["NC"],
                       1,
                       par["NScan"] * par["Nproj"],
                       par["nFE"])) * dcf_coil
    combinedData = FFT.adj(combinedData)
    combinedData = np.fft.fft2(combinedData,
                               norm='ortho')

    for i in range(0, (par["NSlice"])):
        sys.stdout.write(
            "Computing coil sensitivity map of slice %i \r" %
            (i))
        sys.stdout.flush()

        result = nlinvns.nlinvns(
                    np.squeeze(combinedData[:, :, i, ...]),
                    nlinvNewtonSteps,
                    True,
                    nlinvRealConstr)

        par["C"][:, i, :, :] = result[2:, -1, :, :]
        sys.stdout.write("slice %i done \r"
                         % (i))
        sys.stdout.flush()
        if not nlinvRealConstr:
            par["phase_map"][i, :, :] = np.exp(
                1j * np.angle(result[0, -1, :, :]))

            # standardize coil sensitivity propar["file"]s
    sumSqrC = np.sqrt(
        np.sum(
            (par["C"] *
             np.conj(
                par["C"])),
            0))  # 4, 9, 128, 128
    par["InScale"] = sumSqrC
    if par["NC"] == 1:
        par["C"] = sumSqrC
    else:
        par["C"] = par["C"] / np.tile(sumSqrC, (par["NC"], 1, 1, 1))
