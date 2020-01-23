#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module holds the classes for different FFT operators.
"""
import numpy as np
from rrsg_cgreco._helper_fun.calckbkernel import calckbkernel
from rrsg_cgreco._helper_fun.goldcomp import cmp as goldcomp
from abc import ABC, abstractmethod
import itertools


class Operator(ABC):
    """ Abstract base class for linear Operators used inp the optimization.

    This class serves as the base class for all linear operators used inp
    the varous optimization algorithms. it requires to implement a forward
    and backward application inp and out of place.

    Attributes
    ----------
      NScan ():
        Number of total measurements (Scans)
      NC ():
        Number of complex coils
      NSlice ():
        Number ofSlices
      dimX ():
        X dimension of the parameter maps
      dimY ():
        Y dimension of the parameter maps
      N ():
        N number of samples per readout
      Nproj ():
        Number of readouts
      DTYPE (numpy.type):
        The complex value precision. Defaults to complex64
      DTYPE_real (numpy.type):
        The real value precision. Defaults to float32
    """

    def __init__(self, par, DTYPE=np.complex64, DTYPE_real=np.float32):
        """ Operator base constructor.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling pos (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          DTYPE (numpy.type):
            The complex value precision. Defaults to complex64
          DTYPE_real (numpy.type):
            The real value precision. Defaults to float32
        """
        self.NSlice = par["NSlice"]
        self.NScan = par["NScan"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.NSmpl = par["nFE"]
        self.NC = par["NC"]
        self.NProj = par["Nproj"]
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real

    @abstractmethod
    def fwd(self, inp):
        """ Apply operator from parameter space to measurement space.

        Apply the linear operator from parameter space to measurement space
        If streamed operations are used the Numpy.Arrays are replaced
        by Numpy.Array
        This method needs to generate a temporary array and will return it as
        the result.

        Args
        ----
          inp (Numpy.Array):
            The complex parameter space data which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.

        Returns
        -------
          Numpy.Array: A PyOpenCL array containing the result of the
          computation.
        """
        ...

    @abstractmethod
    def adj(self, inp):
        """ Apply operator from measurement space to parameter space.

        Apply the linear operator from measurement space to parameter space.
        If streamed operations are used the Numpy.Arrays are replaced
        by Numpy.Array
        This method needs to generate a temporary array and will return it as
        the result.

        Args
        ----
          inp (Numpy.Array):
            The complex measurement space which is used as input.
          wait_for (list of PyopenCL.Event):
            A List of PyOpenCL events to wait for.

        Returns
        -------
          Numpy.Array: A PyOpenCL array containing the result of the
          computation.
        """
        ...


class NUFFT(Operator):
    """ Non-uniform FFT object.

    This class performs the non-uniform FFT (NUFFT) operation. Linear
    erpolation of a sampled gridding kernel is used to regrid pos
    from the non-cartesian grid back on the cartesian grid.

    Attributes
    ----------
      traj (Numpy.Array):
        The comlex sampling trajectory
      dcf (Numpy.Array):
        The densitiy compenation function
      ogf (float):
        The overgriddingfactor for non-cartesian k-spaces.
      fft_shape (tuple of s):
        3 dimensional tuple. Dim 0 contas all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale (float32):
        The scaling factor to achieve a good adjoness of the forward and
        backward FFT.
      cl_kerneltable (PyOpenCL.Buffer):
        The gridding lookup table as read only Buffer
      cl_deapo (PyOpenCL.Buffer):
        The deapodization lookup table as read only Buffer
      par_fft ():
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft (gpyfft.fft.FFT):
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused inp each iterations, iterationg over
        all scans to keep the memory footpr low.
      prg (PyOpenCL.Program):
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            par,
            kwidth=5,
            fft_dim=(
                1,
                2),
            klength=2000,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        """ NUFFT object constructor.

        Args
        ----
          ctx (PyOpenCL.Context):
            The context for the PyOpenCL computations.
          queue (PyOpenCL.Queue):
            The computation Queue for the PyOpenCL kernels.
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling pos (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          kwidth ():
            The width of the sampling kernel for regridding of non-uniform
            kspace samples.
          klength ():
            The length of the kernel lookup table which samples the contineous
            gridding kernel.
          fft_dim (tuple of s):
            The dimensions which should be transformed. Defaults
            to 1 and 2 corresponding the the last two of fft_dim.
          DTYPE (Numpy.Type):
            The comlex precission type. Currently complex64 is used.
          DTYPE_real (Numpy.Type):
            The real precission type. Currently float32 is used.
        """
        super().__init__(par, DTYPE, DTYPE_real)
        self.ogf = par["nFE"]/par["dimX"]

        (self.kerneltable, kerneltable_FT, u) = calckbkernel(
            kwidth, self.ogf, par["nFE"], klength)

        deapo = 1 / kerneltable_FT.astype(DTYPE_real)
        self.deapo = np.outer(deapo, deapo)

        dcf = np.sqrt(goldcomp(par["traj"]))

        self.dcf = dcf
        self.traj = par["traj"]

        self.nkrnlpts = self.kerneltable.size
        self.gridsize = par["nFE"]
        self.kwidth = (kwidth / 2)/self.gridsize

    def adj(self, inp):
        """ Perform the inverse (adjo) NUFFT operation.

        Args
        ----
          sg (Numpy.Array):
            The complex image data.
          s (Numpy.Array):
            The non-uniformly gridded k-space
        """
        # Grid k-space
        ogkspace = self._grid_lut(inp)
        # FFT
        ogkspace = np.fft.ifftshift(ogkspace, axes=(-2, -1))
        ogkspace = np.fft.ifft2(ogkspace, norm='ortho')
        ogkspace = np.fft.ifftshift(ogkspace, axes=(-2, -1))

        return self._deapo_adj(ogkspace)

    def fwd(self, inp):
        """ Perform the forward NUFFT operation.

        Args
        ----
          s (Numpy.Array):
            The non-uniformly gridded k-space.
          sg (Numpy.Array):
            The complex image data.
        """
        # Deapodization and Scaling
        ogkspace = self._deapo_fwd(inp)
        # FFT
        ogkspace = np.fft.fftshift(ogkspace, axes=(-2, -1))
        ogkspace = np.fft.fft2(ogkspace, norm='ortho')
        ogkspace = np.fft.fftshift(ogkspace, axes=(-2, -1))

        # Resample on Spoke
        return self._invgrid_lut(ogkspace)

    def _deapo_adj(self, inp):
        return inp[...,
                   int(self.gridsize/2-self.dimY/2):
                   int(self.gridsize/2+self.dimY/2),
                   int(self.gridsize/2-self.dimX/2):
                   int(self.gridsize/2+self.dimX/2)]*self.deapo

    def _deapo_fwd(self, inp):
        out = np.zeros((self.NScan, self.NC, self.NSlice,
                        self.gridsize, self.gridsize),
                       dtype=self.DTYPE)
        out[...,
            int(self.gridsize/2-self.dimY/2):
            int(self.gridsize/2+self.dimY/2),
            int(self.gridsize/2-self.dimX/2):
            int(self.gridsize/2+self.dimX/2)] = inp*self.deapo
        return out

    def _grid_lut(self, s):

        gridcenter = self.gridsize/2

        sg = np.zeros((self.NScan, self.NC, self.NSlice,
                       self.gridsize, self.gridsize),
                      dtype=self.DTYPE)

        kdat = s*self.dcf
        for iscan, iproj, ismpl in itertools.product(range(self.NScan),
                                                     range(self.NProj),
                                                     range(self.NSmpl)):
            kx = self.traj[iscan, iproj, ismpl].imag
            ky = self.traj[iscan, iproj, ismpl].real

            ixmin = int((kx-self.kwidth)*self.gridsize + gridcenter)
            ixmax = int((kx+self.kwidth)*self.gridsize + gridcenter) + 1
            iymin = int((ky-self.kwidth)*self.gridsize + gridcenter)
            iymax = int((ky+self.kwidth)*self.gridsize + gridcenter) + 1

            for gcount1 in np.arange(ixmin, ixmax+1):
                dkx = (gcount1-gridcenter) / self.gridsize - kx
                for gcount2 in np.arange(iymin, iymax+1):
                    dky = (gcount2-gridcenter) / self.gridsize - ky
                    dk = np.sqrt(dkx**2+dky**2)
                    if (dk < self.kwidth):

                        fracind = dk/self.kwidth*(self.nkrnlpts-1)
                        kernelind = int(fracind)
                        fracdk = fracind-kernelind

                        kern = self.kerneltable[kernelind]*(1-fracdk) +\
                            self.kerneltable[kernelind+1]*fracdk
                        indx = gcount1
                        indy = gcount2
                        if (gcount1 < 0):
                            indx += self.gridsize
                            indy = self.gridsize-indy
                        if (gcount1 >= self.gridsize):
                            indx -= self.gridsize
                            indy = self.gridsize-indy
                        if (gcount2 < 0):
                            indy += self.gridsize
                            indx = self.gridsize-indx
                        if (gcount2 >= self.gridsize):
                            indy -= self.gridsize
                            indx = self.gridsize-indx

                        sg[iscan, :, :, indy, indx] += \
                            kern * kdat[iscan, :, :, iproj, ismpl]
        return sg

    def _invgrid_lut(self, sg):
        gridcenter = self.gridsize/2

        s = np.zeros((self.NScan, self.NC, self.NSlice,
                      self.NProj, self.NSmpl),
                     dtype=self.DTYPE)

        for iscan, iproj, ismpl in itertools.product(range(self.NScan),
                                                     range(self.NProj),
                                                     range(self.NSmpl)):
            kx = self.traj[iscan, iproj, ismpl].imag
            ky = self.traj[iscan, iproj, ismpl].real

            ixmin = int((kx-self.kwidth)*self.gridsize + gridcenter)
            ixmax = int((kx+self.kwidth)*self.gridsize + gridcenter) + 1
            iymin = int((ky-self.kwidth)*self.gridsize + gridcenter)
            iymax = int((ky+self.kwidth)*self.gridsize + gridcenter) + 1

            for gcount1 in np.arange(ixmin, ixmax+1):
                dkx = (gcount1-gridcenter) / self.gridsize - kx
                for gcount2 in np.arange(iymin, iymax+1):
                    dky = (gcount2-gridcenter) / self.gridsize - ky
                    dk = np.sqrt(dkx**2+dky**2)
                    if (dk < self.kwidth):

                        fracind = dk/self.kwidth*(self.nkrnlpts-1)
                        kernelind = int(fracind)
                        fracdk = fracind-kernelind

                        kern = self.kerneltable[kernelind]*(1-fracdk) +\
                            self.kerneltable[kernelind+1]*fracdk
                        indx = gcount1
                        indy = gcount2
                        if (gcount1 < 0):
                            indx += self.gridsize
                            indy = self.gridsize-indy
                        if (gcount1 >= self.gridsize):
                            indx -= self.gridsize
                            indy = self.gridsize-indy
                        if (gcount2 < 0):
                            indy += self.gridsize
                            indx = self.gridsize-indx
                        if (gcount2 >= self.gridsize):
                            indy -= self.gridsize
                            indx = self.gridsize-indx

                        s[iscan, :, :, iproj, ismpl] += \
                            kern*sg[iscan, :, :, indy, indx]
        return s*self.dcf


class MRIImagingModel(Operator):
    """ The MRI imaging model including Coils.

    TODO

    Attributes
    ----------
      traj (Numpy.Array):
        The comlex sampling trajectory
      dcf (Numpy.Array):
        The densitiy compenation function
      ogf (float):
        The overgriddingfactor for non-cartesian k-spaces.
      fft_shape (tuple of s):
        3 dimensional tuple. Dim 0 contas all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale (float32):
        The scaling factor to achieve a good adjoness of the forward and
        backward FFT.
      cl_kerneltable (PyOpenCL.Buffer):
        The gridding lookup table as read only Buffer
      cl_deapo (PyOpenCL.Buffer):
        The deapodization lookup table as read only Buffer
      par_fft ():
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft (gpyfft.fft.FFT):
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused inp each iterations, iterationg over
        all scans to keep the memory footpr low.
      prg (PyOpenCL.Program):
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            par,
            trajectory,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        """ NUFFT object constructor.

        Args
        ----
          ctx (PyOpenCL.Context):
            The context for the PyOpenCL computations.
          queue (PyOpenCL.Queue):
            The computation Queue for the PyOpenCL kernels.
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling pos (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          kwidth ():
            The width of the sampling kernel for regridding of non-uniform
            kspace samples.
          klength ():
            The length of the kernel lookup table which samples the contineous
            gridding kernel.
          fft_dim (tuple of s):
            The dimensions which should be transformed. Defaults
            to 1 and 2 corresponding the the last two of fft_dim.
          DTYPE (Numpy.Type):
            The comlex precission type. Currently complex64 is used.
          DTYPE_real (Numpy.Type):
            The real precission type. Currently float32 is used.
        """
        super().__init__(par, DTYPE, DTYPE_real)
        par["traj"] = trajectory[None, ...]
        self.NUFFT = NUFFT(par, DTYPE=DTYPE, DTYPE_real=DTYPE_real)
        self.Coils = par["C"]
        self.conjCoils = np.conj(par["C"])

    def adj(self, inp):
        """ Perform the inverse (adjo) NUFFT operation.

        Args
        ----
          sg (Numpy.Array):
            The complex image data.
          s (Numpy.Array):
            The non-uniformly gridded k-space
        """
        return np.sum(self.NUFFT.adj(inp)*self.conjCoils, 1)

    def fwd(self, inp):
        """ Perform the forward NUFFT operation.

        Args
        ----
          s (Numpy.Array):
            The non-uniformly gridded k-space.
          sg (Numpy.Array):
            The complex image data.
        """
        return self.NUFFT.fwd(inp*self.Coils)