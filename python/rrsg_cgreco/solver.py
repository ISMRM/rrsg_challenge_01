#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The CG algorithm for image reconstruction."""
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
        image_dim (int):
            Dimension of the reconstructed image
        num_coils (int):
            Number of coils
        do_incor (bool):
            Flag to do intensity correction:
        incor (np.complex64):
            Intensity correction array
        maxit (int):
            Maximum number of CG iterations
        lambd (float):
            Weight of the Tikhonov regularization. Defaults to 0
        tol (float):
            Tolerance to terminate iterations. Defaults to 0
        NSlice (ind):
           Number of Slices
        DTYPE (numpy.type):
            The complex value precision. Defaults to complex64
        DTYPE_real (numpy.type):
            The real value precision. Defaults to float32
        res (list):
            List to store residual values
        operator (linop.Operator):
            MRI imaging operator to traverse from k-space to imagespace and
            vice versa.

    """

    def __init__(self, data_par, optimizer_par):
        """
        CG object constructor.

        Args
        ----
            data_par (dict):
                A python dict containing the necessary information to
                setup the object. Needs to contain the image dimensions
                (img_dim), number of coils (num_coils),
                sampling points (num_reads) and read outs (num_proj) and the
                complex coil sensitivities (Coils).
            optimizer_par (dict):
                Parameter containing the optimization related settings.

        """
        self.image_dim = data_par["image_dimension"]
        self.num_coils = data_par["num_coils"]
        self.mask = data_par["mask"]

        self.DTYPE = data_par["DTYPE"]
        self.DTYPE_real = data_par["DTYPE_real"]

        self.do_incor = data_par["do_intensity_scale"]
        self.incor = data_par["in_scale"].astype(self.DTYPE)
        self.maxit = optimizer_par["max_iter"]
        self.lambd = optimizer_par["lambda"]
        self.tol = optimizer_par["tolerance"]

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

        This function computes the left hand side of the normal equation,
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

        This function computes the right hand side of the normal equation,
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
        """
        Perform k-space filtering.

        This function filters out k-space points outside the acquired
        trajectory, setting the corresponding k-space position to 0.

        Args
        ----
            x (numpy.Array):
                The complex image-space data to filter.

        Returns
        -------
            numpy.Array: Filtered image-space data.
        """
        print("Performing k-space filtering")
        kpoints = (np.arange(-np.floor(self.operator.grid_size/2),
                             np.ceil(self.operator.grid_size/2))
                   )/self.operator.grid_size
        xx, yy = np.meshgrid(kpoints, kpoints)
        gridmask = np.sqrt(xx**2+yy**2)
        gridmask[np.abs(gridmask) > 0.5] = 1
        gridmask[gridmask < 1] = 0
        gridmask = ~gridmask.astype(bool)
        gridcenter = self.operator.grid_size / 2

        tmp = np.zeros(
            (
                x.shape[0],
                self.operator.grid_size,
                self.operator.grid_size),
            dtype=self.operator.DTYPE
            )
        tmp[
            ...,
            int(gridcenter-self.image_dim/2):
                int(gridcenter+self.image_dim/2),
            int(gridcenter-self.image_dim/2):
                int(gridcenter+self.image_dim/2)
            ] = x

        tmp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
                    tmp, (-2, -1)), norm='ortho'),
                    (-2, -1))*gridmask, (-2, -1)),
                    norm='ortho'), (-2, -1))
        x = tmp[
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
        start = time.perf_counter()
        result = self._cg_solve(
            x=guess,
            data=data,
            iters=self.maxit,
            lambd=self.lambd,
            tol=self.tol
            )
        result[~np.isfinite(result)] = 0
        end = time.perf_counter()-start
        print("\n"+"-"*80)
        print("Elapsed time: %f seconds" % (end))
        print("-"*80)
        if self.do_incor:
            result = self.mask*result/self.incor
        return (self.kspace_filter(result), self.res)

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
                print("\nConverged after %i iterations to %1.3e." %
                      (i+1, delta))
                x[0] = b
                return x[:i+1, ...]
            if not np.mod(i, 1):
                print("Residuum at iter %i : %1.3e" % (i+1, delta), end='\r')

            beta = (np.vdot(residual_new, residual_new) /
                    np.vdot(residual, residual))
            p = residual_new + beta * p
            (residual, residual_new) = (residual_new, residual)
        x[0] = b
        return x
