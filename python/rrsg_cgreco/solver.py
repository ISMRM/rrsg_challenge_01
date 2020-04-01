#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 2019

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
import time


class CGReco:
    """
    Conjugate Gradient Optimization.

    This class performs conjugate gradient based optimization given
    some data and a operator derived from linop.Operator. The operator needs
    to be set using the set_operator method prior to optimization.
    Code is based on the Wikipedia entry
    https://en.wikipedia.org/wiki/Conjugate_gradient_method

    Attributes
    ----------
        dimX (int):
            X dimension of the parameter maps
        dimY (int):
            Y dimension of the parameter maps
        num_coils (int):
            Number of coils
        NScan (int):
            Number of Scans
       NSlice (ind):
           Number of Slices
        DTYPE (numpy.type):
            The complex value precision. Defaults to complex64
        DTYPE_real (numpy.type):
            The real value precision. Defaults to float32
    """

    def __init__(self, data_par, optimizer_par):
        """
        CG object constructor.

        Args
        ----
            par (dict):
                A python dict containing the necessary information to
                setup the object. Needs to contain the number of slices
                (NSlice), number of scans (NScan), image dimensions
                (dimX, dimY), number of coils (num_coils),
                sampling pos (N) and read outs (NProj) and the
                complex coil sensitivities (C).
            DTYPE (Numpy.Type):
                The comlex precission type. Currently complex64 is used.
            DTYPE_real (Numpy.Type):
                The real precission type. Currently float32 is used.
        """
        self.image_dim = data_par["image_dim"]
        self.num_coils = data_par["num_coils"]
        self.DTYPE = data_par["DTYPE"]
        self.DTYPE_real = data_par["DTYPE_real"]

        self.do_incor = data_par["do_intensity_scale"]
        self.incor = data_par["in_scale"].astype(self.DTYPE)
        
        self.maxit=optimizer_par["max_iter"]
        self.lambd=optimizer_par["lambda"]
        self.tol=optimizer_par["tolerance"]

        self.fval_min = 0
        self.fval = 0
        self.res = []
        self.operator = None

    def set_operator(self, op):
        """
        Set the linear operator for CG minimization.

        This functions sets the linear operator to use for CG minimization. It
        is mandatory to set an operator prior to minimization.

        Args
        ----
            op (linop.Operator):
                The linear operator to be used for CG reconstruction.

        """
        self.operator = op

    def operator_lhs(self, inp):
        """
        Compute the LHS of the normal equation.

        This functions compute the left hand side of the normal equation,
        evaluation A^TA x on the input x.

        Args
        ----
            inp (numpy.Array):
               The complex image space data.

        Returns
        -------
            numpy.Array: Left hand side of CG normal equation
        """
        assert self.operator is not None, \
            "Please set an operator with the set_operation method"

        return self.operator_rhs(self.operator.forward(inp))

    def operator_rhs(self, inp):
        """
        Compute the RHS of the normal equation.

        This functions compute the right hand side of the normal equation,
        evaluation A^T b on the k-space data b.

        Args
        ----
            inp (numpy.Array):
                The complex k-space data which is used as input.

        Returns
        -------
            numpy.Array: Right hand side of CG normal equation
        """
        assert self.operator is not None, \
            "Please set an operator with the set_operation method"

        return self.operator.adjoint(inp)

    def kspace_filter(self, x):
        print("Performing k-space filtering")
        kpoints = (np.arange(-np.floor(self.operator.num_reads/2),
                            np.ceil(self.operator.num_reads/2))
                   )/self.operator.num_reads
        xx, yy = np.meshgrid(kpoints,kpoints)
        gridmask = np.sqrt(xx**2+yy**2)
        gridmask[np.abs(gridmask)>0.5] = 1
        gridmask[gridmask<1] = 0
        gridmask = ~gridmask.astype(bool)
        gridcenter = self.operator.num_reads / 2
        for j in range(x.shape[0]):
            tmp = np.zeros(
                (
                    self.operator.num_reads,
                    self.operator.num_reads),
                dtype=self.operator.DTYPE
                )

            tmp[
                ...,
                int(gridcenter-self.image_dim/2):
                    int(gridcenter+self.image_dim/2),
                int(gridcenter-self.image_dim/2):
                    int(gridcenter+self.image_dim/2)
                ] = x[j]

            tmp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
                np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
                        tmp, (-2, -1)), norm='ortho'),
                        (-2, -1))*gridmask, (-2, -1)),
                        norm='ortho'), (-2, -1))
            x[j] = tmp[
                ...,
                int(gridcenter-self.image_dim/2):
                    int(gridcenter+self.image_dim/2),
                int(gridcenter-self.image_dim/2):
                    int(gridcenter+self.image_dim/2)
                ]
        return x

###############################################################################
#   Start a Reconstruction ####################################################
#   Call inner optimization ###################################################
#   output: optimal value of x ################################################
###############################################################################
    def optimize(self, data, guess=None):
        """
        Perform CG minimization.

        Check if operator is set prior to optimization. If no initial guess is
        set, assume all zeros as initial image. Calls the _cg_solve function
        for minimization itself. An optional Tikhonov regularization can be
        enabled which adds a scaled identity matrix to the LHS
        (A^T A + lambd*Id).

        Args
        ----
            data (numpy.Array):
                The complex k-space data to fit.
            guess (numpy.Array=None):
                Optional initial guess for the image.
            maxit (int=10):
                Maximum number of CG steps.
            lambd (float=1e-8):
                Regularization weight for Tikhonov regularizer.
            tol (float=1e-5):
                Relative tolerance to terminate optimization.

        Returns
        -------
            numpy.Array:
                final result of optimization,
                including intermediate results for each CG setp
        """
        if self.operator is None:
            print("Please set an Linear operator "
                  "using the SetOperator method.")
            return

        if guess is None:
            guess = np.zeros(
              (
                  self.maxit+1,
                  self.image_dim,
                  self.image_dim
                  ),
              dtype=self.DTYPE
              )
        start = time.time()
        result = self._cg_solve(
            x=guess,
            data=data,
            iters=self.maxit,
            lambd=self.lambd,
            tol=self.tol
            )
        result[~np.isfinite(result)] = 0
        end = time.time()-start
        print("-"*80)
        print("Elapsed time: %f seconds" % (end))
        print("-"*80)
        if self.do_incor:
            result *= self.incor
        return self.kspace_filter(result)

###############################################################################
#   Conjugate Gradient optimization ###########################################
#   input: initial guess x ####################################################
#          number of iterations iters #########################################
#   output: optimal value of x ################################################
###############################################################################
    def _cg_solve(self, x, data, iters, lambd, tol):
        """
        Conjugate gradient optimization.

        This function implements the CG optimization. It is based on
        https://en.wikipedia.org/wiki/Conjugate_gradient_method

        Args
        ----
            X (numpy.Array):
                Initial guess for the image.
            data (numpy.Array):
                The complex k-space data to fit.
            guess (numpy.Array):
                Optional initial guess for the image.
            iters (int):
                Maximum number of CG steps.
            lambd (float):
                Regularization weight for Tikhonov regularizer.
            tol (float):
                Relative tolerance to terminate optimization.

        Returns
        -------
            numpy.Array:
                A numpy array containing the result of the
                computation.
        """
        b = np.zeros(
            (
                self.image_dim,
                self.image_dim
                ),
            self.DTYPE
            )
        Ax = np.zeros(
            (
                self.image_dim,
                self.image_dim
                ),
            self.DTYPE
            )

        b = self.operator_rhs(data)
        residual = b
        p = residual
        delta = np.linalg.norm(residual) ** 2 / np.linalg.norm(b) ** 2
        self.res.append(delta)
        print("Initial Residuum: ", delta)

        for i in range(iters):
            Ax = self.operator_lhs(p)
            Ax = Ax + lambd * p
            alpha = np.vdot(residual, residual)/(np.vdot(p, Ax))
            x[i + 1] = x[i] + alpha * p
            residual_new = residual - alpha * Ax
            delta = np.linalg.norm(residual_new) ** 2 / np.linalg.norm(b) ** 2
            self.res.append(delta)
            if delta < tol:
                print("Converged after %i iterations to %1.3e." % (i+1, delta))
                return x[:i+1, ...]
            if not np.mod(i, 1):
                print("Residuum at iter %i : %1.3e" % (i+1, delta), end='\r')

            beta = (np.vdot(residual_new, residual_new) /
                    np.vdot(residual, residual))
            p = residual_new + beta * p
            (residual, residual_new) = (residual_new, residual)
        return x
