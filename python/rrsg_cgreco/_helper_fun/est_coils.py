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
import skimage.filters
from scipy.ndimage.morphology import binary_dilation as dilate

# Estimates sensitivities and complex image.
# (see Martin Uecker: Image reconstruction by regularized nonlinear
# inversion joint estimation of coil sensitivities and image content)

def estimate_coil_sensitivities(data, trajectory, par, 
                                coils=None, NLINV=False):
    """
      Estimate complex coil sensitivities.

      Estimate complex coil sensitivities using either a sum-of-squares based
      approach (NLINV=False) or NLINV from Martin Uecker et al. (NLINV=True)


    Args
    ----
          coils (numpy.array):
              Complex coil sensitivites, possibly read from File
          data (numpy.array):
              complex k-space data
          trajectory (numpy.array):
              trajectory information
          par (dict):
              A python dict containing the necessary information to
              setup the object. Needs to contain the number of slices
              (num_slc), number of scans (num_scans),
              image dimensions (dimX, dimY), number of coils (num_coils),
              sampling pos (num_reads) and read outs (num_proj).
          NLINV (bool):
              Switch between NLINV or sum-of-squares based coil estimation.
              Defaults to sum-of-squares (NLINV=False).
    """
    if coils is not None:
        print("Using supplied coil sensitivity profiles...")
        par["Data"]["coils"] = coils
        par["Data"]["phase_map"] = np.zeros(
            (par["Data"]["image_dimension"],
             par["Data"]["image_dimension"]), 
            dtype=par["Data"]["DTYPE"])
        cropfov = slice(int(par["Data"]["coils"].shape[-2]/2
                            -par["Data"]["image_dimension"]/2),
                        int(par["Data"]["coils"].shape[-1]/2
                            +par["Data"]["image_dimension"]/2))
        par["Data"]["coils"] = np.squeeze(
            par["Data"]["coils"][..., cropfov, cropfov])
        if type(par["Data"]["mask"]) is np.ndarray:
            par["Data"]["mask"] = np.squeeze(
            par["Data"]["mask"][cropfov, cropfov])
            
        _norm_coils(par)
    else:    
        if NLINV:
            estimate_coil_sensitivities_NLINV(data, trajectory, par)
        else:
            estimate_coil_sensitivities_SOS(data, trajectory, par)
        
        
def estimate_coil_sensitivities_SOS(data, trajectory, par):   
    """
      Estimate complex coil sensitivities using a sum-of-squares approach.

      Estimate complex coil sensitivities by dividing each coil channel
      with the SoS reconstruciton. A Hann window is used to filter out high
      k-space frequencies.


    Args
    ----
          data (numpy.array):
              complex k-space data
          trajectory (numpy.array):
              trajectory information
          par (dict):
              A python dict containing the necessary information to
              setup the object. Needs to contain the number of slices
              (num_slc), number of scans (num_scans),
              image dimensions (dimX, dimY), number of coils (num_coils),
              sampling pos (num_reads) and read outs (num_proj).
    """
    par["Data"]["phase_map"] = np.zeros(
        (par["Data"]["image_dimension"],
         par["Data"]["image_dimension"]), 
        dtype=par["Data"]["DTYPE"])    

    FFT = linop.NUFFT(par=par, trajectory=trajectory)
    

    windowsize = par["Data"]["num_reads"]/10
    window = np.hanning(windowsize)
    window = np.pad(window, int((par["Data"]["num_reads"]-windowsize)/2))
    
    lowres_data = data*window.T

    coil_images = FFT.adjoint(lowres_data * par["FFT"]["dens_cor"])
    combined_image = np.sqrt(
        1/coil_images.shape[0]
        *np.sum(np.abs(coil_images)**2,0)
        )
    
    coils = coil_images/combined_image
    thresh = skimage.filters.threshold_otsu(combined_image)
    mask = combined_image > thresh*0.3
    mask = dilate(mask, iterations=10)
    

    par["Data"]["coils"] = coils
    par["Data"]["mask"] = mask
    _norm_coils(par)
    
def estimate_coil_sensitivities_NLINV(data, trajectory, par):
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
              trajectory information
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
         par["Data"]["image_dimension"], 
         par["Data"]["image_dimension"]), 
        dtype=par["Data"]["DTYPE"])

    par["Data"]["phase_map"] = np.zeros(
        (par["Data"]["image_dimension"],
         par["Data"]["image_dimension"]), 
        dtype=par["Data"]["DTYPE"])

    FFT = linop.NUFFT(par=par, trajectory=trajectory)

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

    # normalize coil sensitivity profiles
    _norm_coils(par)
            
def _norm_coils(par):
    # normalize coil sensitivity profiles
    sumSqrC = np.sqrt(
        np.sum(
            (par["Data"]["coils"] * np.conj(par["Data"]["coils"])),
            0
            )
        )
    sumSqrC[sumSqrC==0] = 1
    
    par["Data"]["in_scale"] = sumSqrC
    if par["Data"]["num_coils"] == 1:
        par["Data"]["coils"] = sumSqrC
    else:
        par["Data"]["coils"] = (
            par["Data"]["coils"] / sumSqrC)
        