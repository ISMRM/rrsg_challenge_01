#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:26:41 2019

@author: omaier
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
import h5py
import os
from rrsg_cgreco import linop
from rrsg_cgreco._helper_fun.density_compensation \
    import get_density_from_gridding 


DTYPE = np.complex128
DTYPE_real = np.float64


def setupPar(par):
    par["Data"] = {}
    par["Data"]["overgridfactor"] = 2
    par["Data"]["DTYPE"] = DTYPE
    par["Data"]["DTYPE_real"] = DTYPE_real
    
    par["FFT"] = {}
    par["FFT"]["kernelwidth"] = 5
    par["FFT"]["kernellength"] = 5000

    par["Data"]["num_coils"] = 5
    par["Data"]["image_dim"] = 256
    par["Data"]["num_proj"] = 34
    par["Data"]["num_reads"] = 512
    file = h5py.File(
        '.'+os.sep+'python'+os.sep+'test'+os.sep+'smalltest.h5',
        'r'
        )

    par["traj"] = np.array(
        (file['real_traj'][0].astype(DTYPE_real),
         file['imag_traj'][0].astype(DTYPE_real))
        )
    par["traj"] = np.transpose(par["traj"], (1,2,0))

    par["Data"]["coils"] = (
        np.random.randn(
            par["Data"]["num_coils"],
            par["Data"]["image_dim"],
            par["Data"]["image_dim"]
            ) + 1j *
        np.random.randn(
            par["Data"]["num_coils"],
            par["Data"]["image_dim"],
            par["Data"]["image_dim"]
            )
        )

    FFT = linop.NUFFT(data_par=par["Data"], 
                      fft_par=par["FFT"],
                      trajectory=par["traj"])
    par["FFT"]["gridding_matrix"] = FFT.gridding_mat
    par["FFT"]["dens_cor"] = np.sqrt(get_density_from_gridding(
        par["Data"], 
        par["FFT"]["gridding_matrix"]))


class tmpArgs():
    pass


class OperatorKspaceRadial(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = [0]
        parser.use_GPU = True

        par = {}
        setupPar(par)

        self.op = linop.NUFFT(
            par["Data"],
            par["FFT"],
            par["traj"]
            )

        self.opinfwd = (
            np.random.randn(
                par["Data"]["num_coils"],
                par["Data"]["image_dim"],
                par["Data"]["image_dim"]
                )
            + 1j *
            np.random.randn(
                par["Data"]["num_coils"],
                par["Data"]["image_dim"],
                par["Data"]["image_dim"]
                )
            )
        self.opinadj = (
            np.random.randn(
                par["Data"]["num_coils"],
                par["Data"]["num_proj"],
                par["Data"]["num_reads"]
                )
            + 1j *
            np.random.randn(
                par["Data"]["num_coils"],
                par["Data"]["num_proj"],
                par["Data"]["num_reads"]
                )
            )

        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)

    def test_adj_outofplace(self):
        outfwd = self.op.forward(self.opinfwd)
        outadj = self.op.adjoint(self.opinadj)

        a = np.vdot(
                outfwd.flatten(),
                self.opinadj.flatten()
                ) / self.opinadj.size
        b = np.vdot(
                self.opinfwd.flatten(),
                outadj.flatten()
                ) / self.opinadj.size

        print("Adjointness: %.2e+j%.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=12)


class OperatorMRIRadial(unittest.TestCase):
    def setUp(self):
        parser = tmpArgs()
        parser.streamed = False
        parser.devices = [0]
        parser.use_GPU = True

        par = {}
        setupPar(par)

        self.op = linop.MRIImagingModel(
            par,
            par["traj"]
            )

        self.opinfwd = (
            np.random.randn(
                par["Data"]["image_dim"],
                par["Data"]["image_dim"]
                )
            + 1j *
            np.random.randn(
                par["Data"]["image_dim"],
                par["Data"]["image_dim"]
                )
            )
        self.opinadj = (
            np.random.randn(
                par["Data"]["num_coils"],
                par["Data"]["num_proj"],
                par["Data"]["num_reads"]
                )
            + 1j *
            np.random.randn(
                par["Data"]["num_coils"],
                par["Data"]["num_proj"],
                par["Data"]["num_reads"]
                )
            )

        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)

    def test_adj_outofplace(self):
        outfwd = self.op.forward(self.opinfwd)
        outadj = self.op.adjoint(self.opinadj)

        a = np.vdot(
                outfwd.flatten(),
                self.opinadj.flatten()
                ) / self.opinadj.size
        b = np.vdot(
                self.opinfwd.flatten(),
                outadj.flatten()
                ) / self.opinadj.size

        print("Adjointness: %.2e+j%.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=12)
