#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Linear operators for MRI image reconstruction."""
import numpy as np
from rrsg_cgreco._helper_fun.calckbkernel import calculate_keiser_bessel_kernel
from rrsg_cgreco._helper_fun.goldcomp import get_golden_angle_dcf as goldcomp
from abc import ABC, abstractmethod
import itertools


class Operator(ABC):
    """
    Abstract base class for linear Operators used in the optimization.

    This class serves as the base class for all linear operators used inp
    the varous optimization algorithms. it requires to implement a forward
    and backward application inp and out of place.

    Attributes
    ----------
      num_scans (int):
        Number of total measurements (Scans)
      num_coils (int):
        Number of complex coils
      num_slc (int):
        Number ofSlices
      dimX (int):
        X dimension of the parameter maps
      dimY (int):
        Y dimension of the parameter maps
      N (int):
        N number of samples per readout
      num_proj (int):
        Number of readouts
      DTYPE (numpy.type):
        The complex value precision. Defaults to complex64
      DTYPE_real (numpy.type):
        The real value precision. Defaults to float32
    """

    def __init__(self, par, DTYPE=np.complex64, DTYPE_real=np.float32):
        """
        Operator base constructor.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (num_slc),
            number of scans (num_scans), image dimensions (dimX, dimY),
            number of coils (num_coils),
            sampling pos (N) and read outs (num_proj).
          DTYPE (numpy.type):
            The complex value precision. Defaults to complex64
          DTYPE_real (numpy.type):
            The real value precision. Defaults to float32
        """
        self.num_slc = par["num_slc"]
        self.num_scans = par["num_scans"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.num_reads = par["num_reads"]
        self.num_coils = par["num_coils"]
        self.num_proj = par["num_proj"]
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real

    @abstractmethod
    def fwd(self, inp):
        """
        Apply operator from parameter space to measurement space.

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
        """
        Apply operator from measurement space to parameter space.

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
    """
    Non-uniform FFT object.

    This class performs the non-uniform FFT (NUFFT) operation. Linear
    erpolation of a sampled gridding kernel is used to regrid pos
    from the non-cartesian grid back on the cartesian grid.

    Attributes
    ----------
      traj (Numpy.Array):
        The comlex sampling trajectory
      dens_cor (Numpy.Array):
        The densitiy compenation function
      ogf (float):
        The overgriddingfactor for non-cartesian k-spaces.
      fft_scale (float32):
        The scaling factor to achieve a good adjoness of the forward and
        backward FFT.
      kerneltable (Numpy.Array):
        The gridding lookup table
      deapo (Numpy.Array):
        The deapodization lookup table
      nkrnlpts (int):
        The number of points in the precomputed gridding kernel
      gridsize (int):
        The size of the grid to interpolate to
      kwidth (float):
        The half width of the kernel relative to the number of gridpoints
    """

    def __init__(
            self,
            par,
            trajectory,
            kwidth=5,
            fft_dim=(
                1,
                2),
            klength=2000,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        """
        NUFFT object constructor.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (num_slc),
            number of scans (num_scans), image dimensions (dimX, dimY),
            number of
            coils (num_coils), sampling pos (N) and read outs (num_proj).
          trajectory (numpy.array):
            Complex trajectory information for kx/ky points.
          kwidth (int):
            The width of the sampling kernel for regridding of non-uniform
            kspace samples.
          klength (int):
            The length of the kernel lookup table which samples the contineous
            gridding kernel.
          DTYPE (Numpy.Type):
            The comlex precission type. Currently complex64 is used.
          DTYPE_real (Numpy.Type):
            The real precission type. Currently float32 is used.
        """
        super().__init__(par, DTYPE, DTYPE_real)
        self.ogf = par["num_reads"]/par["dimX"]

        (self.kerneltable, kerneltable_FT, u) = calculate_keiser_bessel_kernel(
            kwidth, self.ogf, par["num_reads"], klength)

        deapo = 1 / kerneltable_FT.astype(DTYPE_real)
        self.deapo = np.outer(deapo, deapo)

        dens_cor = np.sqrt(goldcomp(trajectory))

        self.dens_cor = dens_cor
        self.traj = trajectory

        self.nkrnlpts = self.kerneltable.size
        self.gridsize = par["num_reads"]
        self.kwidth = (kwidth / 2)/self.gridsize

    def adj(self, inp):
        """
        Perform the adjoint (inverse) NUFFT operation.

        This functions performs the adjoint NUFFT operation. It consists of
        gridding of the non-uniform sampled data to the cartesian grid followed
        by a 2D fft, implemented in numpy and final deapodization to
        compensate for the convolution of the gridding kernel on the data.

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
        """
        Perform the forward NUFFT operation.

        This functions performs the forward NUFFT operation. It consists of
        applying the deapodization followed by fourier transofrmation and
        finally resampling of the cartesian data to the non-uniform trajecotry.

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
        out = np.zeros((self.num_scans, self.num_coils, self.num_slc,
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

        sg = np.zeros((self.num_scans, self.num_coils, self.num_slc,
                       self.gridsize, self.gridsize),
                      dtype=self.DTYPE)

        kdat = s*self.dens_cor
        for iscan, iproj, ismpl in itertools.product(range(self.num_scans),
                                                     range(self.num_proj),
                                                     range(self.num_reads)):
            kx = self.traj[iscan, iproj, ismpl].imag
            ky = self.traj[iscan, iproj, ismpl].real

            ixmin = int((kx - self.kwidth) * self.gridsize + gridcenter)
            ixmax = int((kx + self.kwidth) * self.gridsize + gridcenter) + 1
            iymin = int((ky - self.kwidth) * self.gridsize + gridcenter)
            iymax = int((ky + self.kwidth) * self.gridsize + gridcenter) + 1

            for gcount1 in np.arange(ixmin, ixmax+1):
                dkx = (gcount1 - gridcenter) / self.gridsize - kx

                for gcount2 in np.arange(iymin, iymax+1):
                    dky = (gcount2 - gridcenter) / self.gridsize - ky
                    dk = np.sqrt(dkx ** 2 + dky ** 2)

                    if dk < self.kwidth:
                        fracind = dk / self.kwidth * (self.nkrnlpts - 1)
                        kernelind = int(fracind)
                        fracdk = fracind - kernelind

                        kern = self.kerneltable[kernelind] * (1 - fracdk) + self.kerneltable[kernelind + 1] * fracdk
                        indx = gcount1
                        indy = gcount2

                        if gcount1 < 0:
                            indx += self.gridsize
                            indy = self.gridsize - indy

                        if gcount1 >= self.gridsize:
                            indx -= self.gridsize
                            indy = self.gridsize - indy

                        if gcount2 < 0:
                            indy += self.gridsize
                            indx = self.gridsize - indx

                        if gcount2 >= self.gridsize:
                            indy -= self.gridsize
                            indx = self.gridsize - indx

                        sg[iscan, :, :, indy, indx] += kern * kdat[iscan, :, :, iproj, ismpl]
        return sg

    def _invgrid_lut(self, sg):
        gridcenter = self.gridsize/2

        s = np.zeros((self.num_scans, self.num_coils, self.num_slc,
                      self.num_proj, self.num_reads),
                     dtype=self.DTYPE)

        for iscan, iproj, ismpl in itertools.product(range(self.num_scans),
                                                     range(self.num_proj),
                                                     range(self.num_reads)):
            kx = self.traj[iscan, iproj, ismpl].imag
            ky = self.traj[iscan, iproj, ismpl].real

            ixmin = int((kx-self.kwidth)*self.gridsize + gridcenter)
            ixmax = int((kx+self.kwidth)*self.gridsize + gridcenter) + 1
            iymin = int((ky-self.kwidth)*self.gridsize + gridcenter)
            iymax = int((ky+self.kwidth)*self.gridsize + gridcenter) + 1

            for gcount1 in np.arange(ixmin, ixmax+1):
                dkx = (gcount1 - gridcenter) / self.gridsize - kx

                for gcount2 in np.arange(iymin, iymax+1):
                    dky = (gcount2 - gridcenter) / self.gridsize - ky
                    dk = np.sqrt(dkx ** 2 + dky ** 2)
                    if dk < self.kwidth:

                        fracind = dk / self.kwidth * (self.nkrnlpts - 1)
                        kernelind = int(fracind)
                        fracdk = fracind - kernelind

                        kern = self.kerneltable[kernelind] * (1 - fracdk) + self.kerneltable[kernelind + 1] * fracdk
                        indx = gcount1
                        indy = gcount2

                        if gcount1 < 0:
                            indx += self.gridsize
                            indy = self.gridsize - indy

                        if gcount1 >= self.gridsize:
                            indx -= self.gridsize
                            indy = self.gridsize - indy

                        if gcount2 < 0:
                            indy += self.gridsize
                            indx = self.gridsize - indx

                        if gcount2 >= self.gridsize:
                            indy -= self.gridsize
                            indx = self.gridsize - indx

                        s[iscan, :, :, iproj, ismpl] += \
                            kern*sg[iscan, :, :, indy, indx]
        return s*self.dens_cor


class MRIImagingModel(Operator):
    """
    The MRI imaging model including Coils.

    Linear Operator including coil sensitivities and fourier transform to
    transverse between k-space and image space.

    Attributes
    ----------
      coils (Numpy.Array):
        The comlex coil sensitivity profiles
      conj_coils (Numpy.Array):
        The precomputed comlex conjugate coil sensitivity profiles
      NUFFT (linop.NUFFT):
        The NUFFT object to perform gridding.
    """

    def __init__(
            self,
            par,
            trajectory,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        """
        NUFFT object constructor.

        Args
        ----
          par (dict): A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (num_slc),
            number of scans (num_scans), image dimensions (dimX, dimY),
            number of coils (num_coils), sampling pos (N)
            and read outs (num_proj) and the complex
            coil sensitivities (C).
          trajectory (numpy.array):
            Complex trajectory information for kx/ky points.
          DTYPE (Numpy.Type):
            The comlex precission type. Currently complex64 is used.
          DTYPE_real (Numpy.Type):
            The real precission type. Currently float32 is used.
        """
        super().__init__(par, DTYPE, DTYPE_real)

        self.NUFFT = NUFFT(par, trajectory,
                           DTYPE=DTYPE, DTYPE_real=DTYPE_real)
        self.coils = par["coils"]
        self.conj_coils = np.conj(par["coils"])

    def adj(self, inp):
        """
        Perform the adjoint imaging operation.

        Perform the adjoint imaging operation consisting of the NUFFT
        followed by multiplication with the complex conjugate coil
        sensitivity profiles.

        Args
        ----
          sg (Numpy.Array):
            The complex image data.
          s (Numpy.Array):
            The non-uniformly gridded k-space
        """
        return np.sum(self.NUFFT.adj(inp)*self.conj_coils, 1)

    def fwd(self, inp):
        """
        Perform the forward imaging operation.

        Perform the forward imaging operation consisting of multiplication
        with coil sensitivity profiles followed by the NUFFT.

        Args
        ----
          s (Numpy.Array):
            The non-uniformly gridded k-space.
          sg (Numpy.Array):
            The complex image data.
        """
        return self.NUFFT.fwd(inp*self.coils)
