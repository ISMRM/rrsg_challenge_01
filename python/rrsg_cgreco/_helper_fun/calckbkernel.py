#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:05:45 2018

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

import warnings
import numpy as np
from rrsg_cgreco._helper_fun.kb import kaiser_bessel


def calculate_keiser_bessel_kernel(
        kernelwidth, 
        overgridfactor, 
        gridsize, 
        kernellength=32, 
        **kwargs):
    """
    Calculate the appropriate Kaiser-Bessel gridding kernel.

    Function calculates the appropriate Kaiser-Bessel kernel for gridding,
    using the approach of Jackson et al. Original Skript by B. Hargreaves
    Adapted for Python by O. Maier

    Args
    ----
        kernelwidth (int):
            kernel width
        overgridfactor (float):
            over-gridding factor.
        gridsize (int):
            gridsize of oversampled grid
        kernellength (int):
            kernel look-up-table length.

    Returns
    -------
        kern:
            kernel values for kernellength values of u,
            uniformly spaced from 0 to kernelwidth/2.
        kern_ft:
            normalized Fourier transform of kernel used for deapodization
        kbu:
            position of sampling points of kernel.
            linspace from 0 to length of kernel

    """
    if kernellength < 2:
        kernellength = 2
        warnings.warn(
            'Warning:  kernellength must be 2 or more. Default to 2.'
            )

    # From Beatty et al. -
    # Rapid Gridding Reconstruction With a Minimal Oversampling Ratio -
    # equation [5]
    beta = np.pi * np.sqrt(
        (kernelwidth / overgridfactor) ** 2 
        * (overgridfactor - 0.5) ** 2 
        - 0.8
        )
    # Kernel radii.
    u = np.linspace(
        0, 
        (kernelwidth/2), 
        int(np.ceil(kernellength*kernelwidth/2)))

    kern = kaiser_bessel(u, kernelwidth, beta, gridsize)
    kern = kern / kern[u == 0]  # Normalize.

    ft_y = np.flip(kern)
    if np.mod(kernelwidth, 2):
        ft_y = np.concatenate((ft_y[:-1], kern))
    else:
        ft_y = np.concatenate((ft_y, kern))
    
    ft_y = np.pad(
        ft_y, 
        (int(np.floor((gridsize * kernellength - ft_y.size) / 2)),
         int(np.ceil((gridsize * kernellength - ft_y.size) / 2))),
        'constant'
        )
    ft_y = np.abs(
      np.fft.fftshift(
          np.fft.ifft(
              np.fft.ifftshift(
                  ft_y
                  )
              )
          )
      * ft_y.size
      )

    x = np.linspace(
        -int(np.floor(gridsize/(2*overgridfactor))), 
        int(np.floor(gridsize/(2*overgridfactor)))-1, 
        int(np.floor(gridsize/(overgridfactor)))
        )

    ft_y = ft_y[(ft_y.size / 2 + x).astype(int)]
    h = np.sinc(x / (gridsize * kernellength)) ** 2

    kern_ft = ft_y * h
    kern_ft = kern_ft / np.max(kern_ft)

    return kern, kern_ft, u


if __name__ == "__main__":
    # A simple example of what this function does..
    import matplotlib.pyplot as plt
    kernellength = 50
    kernelwidth = 1.5
    for overgridfactor in range(1, 10):
        for G in range(2, 10):
            try:
                a, b, c = calculate_keiser_bessel_kernel(
                    kernelwidth=kernelwidth,
                    gridsize=G,
                    overgridfactor=overgridfactor,
                    kernellength=kernellength
                    )
                plt.plot(
                    a,
                    label='overgridfactor={}, G={}'.format(overgridfactor, G)
                    )
            except ValueError:
                continue
