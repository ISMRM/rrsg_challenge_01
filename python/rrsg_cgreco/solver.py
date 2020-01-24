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
    def __init__(self, par, DTYPE=np.complex64, DTYPE_real=np.float32):
        self.C = par["C"]
        self.traj = par["traj"]
        self.NSlice = par["NSlice"]
        self.NScan = par["NScan"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.NC = par["NC"]
        self.fval_min = 0
        self.fval = 0
        self.res = []
        self.N = par["nFE"]
        self.Nproj = par["Nproj"]
        self.incor = par["InScale"].astype(DTYPE)
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real

        self.operator = None

    def setOperator(self, op):
        self.operator = op

    def operator_lhs(self, inp):
        return self.operator_rhs(self.operator.fwd(inp[None, ...]))

    def operator_rhs(self, inp):
        return self.operator.adj(inp)

###############################################################################
#   Start a Reconstruction ####################################################
#   Call inner optimization ###################################################
#   output: optimal value of x ################################################
###############################################################################
    def optimize(self, data, guess=None, maxit=10, lambd=1e-8, tol=1e-5):
        if self.operator is None:
            print("Please set an Linear operator "
                  "using the SetOperator method.")
            return
        if guess is None:
            guess = np.zeros(
              (maxit+1, 1, 1, self.NSlice, self.dimY, self.dimX),
              dtype=self.DTYPE)
        start = time.time()
        result = self.cg_solve(
          guess, data[None, :, None, ...], maxit, lambd, tol)
        result[~np.isfinite(result)] = 0
        end = time.time()-start
        print("-"*80)
        print("Elapsed time: %f seconds" % (end))
        print("-"*80)
        print("done")
        return result

###############################################################################
#   Conjugate Gradient optimization ###########################################
#   input: initial guess x ####################################################
#          number of iterations iters #########################################
#   output: optimal value of x ################################################
###############################################################################
    def cg_solve(self, x, data, iters, lambd, tol):

        b = np.zeros((self.NScan, 1, self.NSlice, self.dimY, self.dimX),
                     self.DTYPE)
        Ax = np.zeros((self.NScan, 1, self.NSlice, self.dimY, self.dimX),
                      self.DTYPE)

        b = self.operator_rhs(data)
        res = b
        p = res
        delta = np.linalg.norm(res)**2/np.linalg.norm(b)**2
        self.res.append(delta)
        print("Initial Residuum: ", delta)

        for i in range(iters):
            Ax = self.operator_lhs(p)
            Ax = Ax + lambd*p
            alpha = (np.vdot(res, res)/(np.vdot(p, Ax)))
            x[i+1] = (x[i] + alpha*p)
            res_new = res - alpha*Ax
            delta = np.linalg.norm(res_new)**2/np.linalg.norm(b)**2
            self.res.append(delta)
            if delta < tol:
                print("Converged after %i iterations to %1.3e." % (i+1, delta))
                return x[:i+1, ...]
            if not np.mod(i, 1):
                print("Residuum at iter %i : %1.3e" % (i+1, delta), end='\r')

            beta = (np.vdot(res_new, res_new) /
                    np.vdot(res, res))
            p = res_new+beta*p
            (res, res_new) = (res_new, res)
        return x
