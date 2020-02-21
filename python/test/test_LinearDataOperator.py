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
from python.rrsg_cgreco import linop

DTYPE = np.complex128
DTYPE_real = np.float64


def setupPar(par):
    par["NScan"] = 1
    par["NC"] = 5
    par["NSlice"] = 1
    par["dimX"] = 256
    par["dimY"] = 256
    par["Nproj"] = 34
    par["N"] = 512
    file = h5py.File('./test/smalltest.h5')

    par["traj"] = file['real_traj'][()].astype(DTYPE) + \
        1j*file['imag_traj'][()].astype(DTYPE)


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
            par,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

        self.opinfwd = np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                 par["dimY"], par["dimX"])*0
        self.opinadj = np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                       par["Nproj"], par["N"]) +\
            1j * np.random.randn(par["NScan"], par["NC"], par["NSlice"],
                                 par["Nproj"], par["N"])*0

        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)

    def test_adj_outofplace(self):
        outfwd = self.op.forward(self.opinfwd)
        outadj = self.op.adj(self.opinadj)

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e+j%.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=12)
