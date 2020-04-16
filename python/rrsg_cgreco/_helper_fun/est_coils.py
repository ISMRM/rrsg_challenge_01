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
from rrsg_cgreco._helper_fun import nlinvns
from rrsg_cgreco import linop

# Estimates sensitivities and complex image.
# (see Martin Uecker: Image reconstruction by regularized nonlinear
# inversion joint estimation of coil sensitivities and image content)


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
              par (dict):
                  A python dict containing the necessary information to
                  setup the object. Needs to contain the number of slices
                  (num_slc), number of scans (num_scans),
                  image dimensions (dimX, dimY), number of coils (num_coils),
                  sampling pos (num_reads) and read outs (num_proj).
    """
    nlinv_newton_steps = 6
    nlinv_real_constr = False

    par["Data"]["coils"] = np.zeros(
        (par["Data"]["num_coils"], 
         par["Data"]["image_dim"], 
         par["Data"]["image_dim"]), 
        dtype=par["Data"]["DTYPE"])

    par["Data"]["phase_map"] = np.zeros(
        (par["Data"]["image_dim"],
         par["Data"]["image_dim"]), 
        dtype=par["Data"]["DTYPE"])

    FFT = linop.NUFFT(data_par=par["Data"], 
                      fft_par=par["FFT"],
                      trajectory=trajectory)

    combined_data = FFT.adjoint(data * par["FFT"]["dens_cor"])
    combined_data = np.fft.fft2(
        combined_data,
        norm='ortho')


    sys.stdout.write(
        "Computing coil sensitivity map\n")
    sys.stdout.flush()

    result = nlinvns.nlinvns(
                np.squeeze(combined_data),
                nlinv_newton_steps,
                True,
                nlinv_real_constr
                )

    par["Data"]["coils"] = result[2:, -1]

    if not nlinv_real_constr:
        par["Data"]["phase_map"] = np.exp(
            1j
            *
            np.angle(
                result[0, -1]
                )
            )

    # standardize coil sensitivity profiles
    sumSqrC = np.sqrt(
        np.sum(
            (par["Data"]["coils"]
             *
             np.conj(
                 par["Data"]["coils"]
                 )
             ),
            0
            )
        )
    par["Data"]["in_scale"] = sumSqrC
    if par["Data"]["num_coils"] == 1:
        par["Data"]["coils"] = sumSqrC
    else:
        par["Data"]["coils"] = (
            par["Data"]["coils"] / sumSqrC)
            
