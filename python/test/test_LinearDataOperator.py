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
from rrsg_cgreco import linop
from rrsg_cgreco._helper_fun import goldcomp as goldcomp

DTYPE = np.complex128
DTYPE_real = np.float64


def setupPar(par):
    par["num_scans"] = 1
    par["num_coils"] = 5
    par["num_slc"] = 1
    par["dimX"] = 256
    par["dimY"] = 256
    par["num_proj"] = 34
    par["num_reads"] = 512
    file = h5py.File('./python/test/smalltest.h5')

    par["traj"] = (
        file['real_traj'][()].astype(DTYPE) + 1j *
        file['imag_traj'][()].astype(DTYPE)
        )

    par["coils"] = (
        np.random.randn(
            par["num_coils"],
            par["num_slc"],
            par["dimY"],
            par["dimX"]
            ) + 1j *
        np.random.randn(
            par["num_coils"],
            par["num_slc"],
            par["dimY"],
            par["dimX"]
            )
        )

    par["dens_cor"] = (
        np.sqrt(
            np.array(
                goldcomp.get_golden_angle_dcf(
                         par["traj"]
                         ),
                dtype=DTYPE_real
                )
            )
        .astype(DTYPE_real)
        )
    par["dens_cor"] = np.require(
        np.abs(
            par["dens_cor"]
            ),
        DTYPE_real,
        requirements='C'
        )


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
            par["traj"],
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real
            )

        self.opinfwd = (
            np.random.randn(
                par["num_scans"],
                par["num_coils"],
                par["num_slc"],
                par["dimY"],
                par["dimX"]
                )
            + 1j *
            np.random.randn(
                par["num_scans"],
                par["num_coils"],
                par["num_slc"],
                par["dimY"],
                par["dimX"]
                )
            )
        self.opinadj = (
            np.random.randn(
                par["num_scans"],
                par["num_coils"],
                par["num_slc"],
                par["num_proj"],
                par["num_reads"]
                )
            + 1j *
            np.random.randn(
                par["num_scans"],
                par["num_coils"],
                par["num_slc"],
                par["num_proj"],
                par["num_reads"]
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
            par["traj"],
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real)

        self.opinfwd = (
            np.random.randn(
                par["num_scans"],
                par["num_slc"],
                par["dimY"],
                par["dimX"]
                )
            + 1j *
            np.random.randn(
                par["num_scans"],
                par["num_slc"],
                par["dimY"],
                par["dimX"]
                )
            )
        self.opinadj = (
            np.random.randn(
                par["num_scans"],
                par["num_coils"],
                par["num_slc"],
                par["num_proj"],
                par["num_reads"]
                )
            + 1j *
            np.random.randn(
                par["num_scans"],
                par["num_coils"],
                par["num_slc"],
                par["num_proj"],
                par["num_reads"]
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
