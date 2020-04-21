#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Linear operators for MRI image reconstruction."""

import numpy as np
from rrsg_cgreco._helper_fun.calckbkernel import calculate_keiser_bessel_kernel
from abc import ABC, abstractmethod
import itertools
import scipy.sparse


class Operator(ABC):
    """
    Abstract base class for linear Operators used in the optimization.

    This class serves as the base class for all linear operators used inp
    the various optimization algorithms. it requires to implement a forward
    and backward application in and out of place.

    Attributes
    ----------
        num_scans (int):
            Number of total measurements (Scans)
        num_coils (int):
            Number of complex coils
        num_slc (int):
            Number of slices
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

    def __init__(self, par):
        """
        Operator base constructor.

        Args
        ----
            par (dict):
                A python dict containing the necessary information to
                setup the object. Needs to contain the number of slices
                (num_slc),
                number of scans (num_scans), image dimensions (dimX, dimY),
                number of coils (num_coils),
                sampling pos (N) and read outs (num_proj).
                (Optional) content of the par (dict) is the DTYPE and
                DTYPE_real specification. The default value will be
                np.complex64 and np.float32 respectively.
        """
        # Test whether the given dictionary contains the desired keys
        necessary_keys = ['num_slc', 'num_scans', 'dimX', 'dimY',
                          'num_coils', 'N', 'num_proj']
        necessary_test = [x in par for x in necessary_keys]
        if not all(necessary_test):
            false_index = [i for i, x in enumerate(necessary_test) if x == False]
            missing_keys = [necessary_keys[i] for i in false_index]
            raise ValueError('Missing keys ', missing_keys)

        self.image_dim = par["image_dim"]
        self.num_reads = par["num_reads"]
        self.num_coils = par["num_coils"]
        self.num_proj = par["num_proj"]
        self.DTYPE = par.get("DTYPE", np.complex64)
        self.DTYPE_real = par.get("DTYPE_real", np.float32)

    @abstractmethod
    def forward(self, inp):
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
              Numpy.Array:
                  A PyOpenCL array containing the result of the
                  computation.
        """
        raise NotImplementedError

    @abstractmethod
    def adjoint(self, inp):
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
            Numpy.Array:
                A PyOpenCL array containing the result of the
                computation.
        """
        raise NotImplementedError


class NUFFT(Operator):
    """
    Non-uniform FFT object.

    This class performs the non-uniform FFT (NUFFT) operation. Linear
    interpolation of a sampled gridding kernel is used to re-grid pos
    from the non-cartesian grid back on the cartesian grid.

    Attributes
    ----------
        trajectory (Numpy.Array):
            The complex sampling trajectory
        dens_comp (Numpy.Array):
            The density compensation function
        overgridfactor (float):
            The over-gridding factor for non-cartesian k-spaces.
        fft_scale (float32):
            The scaling factor to achieve a good adjointness of the forward and
            backward FFT.
        kerneltable (Numpy.Array):
            The gridding lookup table
        deapo (Numpy.Array):
            The de-apodization lookup table
        n_kernel_points (int):
            The number of points in the precomputed gridding kernel
        grid_size (int):
            The size of the grid to interpolate to
        kwidth (float):
            The half width of the kernel relative to the number of grid-points
    """

    def __init__(
            self,
            data_par,
            fft_par,
            trajectory,
            fft_dim=(-2, -1),
            ):
        """
        NUFFT object constructor.

        Args
        ----
            data_par (dict):
                A python dict containing the necessary information to
                setup the object. Needs to contain the number of slices
                (num_slc), number of scans (num_scans),
                image dimensions (dimX, dimY), number of coils (num_coils),
                sampling pos (N) and read outs (num_proj).
                (Optional) The dictionary can contain specification of
                DTYPE and DTYPE_real.
                overgridfactor..?? This is not necessary information?
            fft_par (??):
                ... kernelwidth..? kernellength..? ..
                (Optional) gridding_matrix ? dens_cor..? .. Needs to check this content as well..?
                The width of the sampling kernel for re-gridding of non-uniform kspace samples.

                The length of the kernel lookup table which samples the continuous gridding kernel.
            trajectory (numpy.array):
                Trajectory information for kx/ky/kz points. Expects a shape of (num_scans, num_proj, 3)
            fft_dim (tuple):
                A tuple containing the axes over which the Fourier Transform
                is performed.
        """
        super().__init__(data_par)
        
        self.overgridfactor = data_par["overgridfactor"]

        (self.kerneltable, kerneltable_FT, u) = calculate_keiser_bessel_kernel(
            fft_par["kernelwidth"],
            self.overgridfactor,
            data_par["num_reads"],
            fft_par["kernellength"])

        deapodization = 1 / kerneltable_FT.astype(data_par["DTYPE_real"])
        self.deapodization = np.outer(deapodization, deapodization)

        if "dens_cor" in fft_par.keys():
            self.dens_comp = fft_par["dens_cor"]        
        else:
            self.dens_comp = np.ones(
                trajectory.shape[:-1],
                dtype=data_par["DTYPE_real"]
                )
        self.trajectory = trajectory

        self.n_kernel_points = self.kerneltable.size
        self.grid_size = data_par["num_reads"]
        self.kwidth = (fft_par["kernelwidth"] / 2) / self.grid_size
        self.fft_dim = fft_dim
        if "gridding_matrix" in fft_par.keys():
            self.gridding_mat = fft_par["gridding_matrix"]
        else:
            self.gridding_mat = self._generate_gridding_matrix()
        self.gridding_mat_adj = self.gridding_mat.transpose()

    def adjoint(self, inp):
        """
        Perform the adjoint (inverse) NUFFT operation.

        This functions performs the adjoint NUFFT operation. It consists of
        gridding of the non-uniform sampled data to the cartesian grid followed
        by a 2D fft, implemented in numpy and final deapodization to
        compensate for the convolution of the gridding kernel on the data.

        Args:
            inp:

        Returns:

        """
        # Perform density compensation
        denscor_inp = inp * self.dens_comp
        # Grid k-space
        ogkspace = np.zeros(
            (self.num_coils, self.grid_size, self.grid_size), 
            dtype=self.DTYPE)
        for nc in range(self.num_coils):
            ogkspace[nc] = np.reshape(
                self.gridding_mat_adj.dot(denscor_inp[nc].flatten()), 
                (self.grid_size, self.grid_size)
                )

        # FFT
        ogkspace = np.fft.ifftshift(ogkspace, axes=self.fft_dim)
        ogkspace = np.fft.ifft2(ogkspace, norm='ortho')
        ogkspace = np.fft.ifftshift(ogkspace, axes=self.fft_dim)
        # Deapodization and Scaling
        return self._deapo_adj(ogkspace)

    def forward(self, inp):
        """
        Perform the forward NUFFT operation.

        This functions performs the forward NUFFT operation. It consists of
        applying the deapodization followed by fourier transformation and
        finally resampling of the cartesian data to the non-uniform trajectory.

        Args:
            inp:

        Returns:

        """
        # Deapodization and Scaling
        ogkspace = self._deapo_fwd(inp)
        # FFT
        ogkspace = np.fft.fftshift(ogkspace, axes=(-2, -1))
        ogkspace = np.fft.fft2(ogkspace, norm='ortho')
        ogkspace = np.fft.fftshift(ogkspace, axes=(-2, -1))

        # Resample on Spoke
        kspace = np.zeros((self.num_coils, self.num_proj, self.num_reads), dtype=self.DTYPE)
        for nc in range(self.num_coils):
            kspace[nc] = np.reshape(
                self.gridding_mat.dot(ogkspace[nc].flatten()), 
                (self.num_proj, self.num_reads)
                )
        # Perform density compensation
        return kspace*self.dens_comp

    def _deapo_adj(self, inp):
        gridcenter = self.grid_size / 2

        return inp[
            ...,
            int(gridcenter-self.image_dim/2):
                int(gridcenter+self.image_dim/2),
            int(gridcenter-self.image_dim/2):
                int(gridcenter+self.image_dim/2)
            ] * self.deapodization

    def _deapo_fwd(self, inp):
        gridcenter = self.grid_size / 2

        out = np.zeros(
            (
                self.num_coils,
                self.grid_size,
                self.grid_size),
            dtype=self.DTYPE
            )

        out[
            ...,
            int(gridcenter-self.image_dim/2):
                int(gridcenter+self.image_dim/2),
            int(gridcenter-self.image_dim/2):
                int(gridcenter+self.image_dim/2)
            ] = inp * self.deapodization
        return out

    def _grid_lut(self, s, return_mapping=False):
        gridcenter = self.grid_size / 2

        sg = np.zeros(
            (
                self.num_coils,
                self.grid_size,
                self.grid_size
                ),
            dtype=self.DTYPE
            )
        grid_point_mapping = []  # Here for demonstration purposes.

        kdat = s * self.dens_comp
        for iproj, iread in itertools.product(
                range(self.num_proj),
                range(self.num_reads)
                ):

            temp_mapping = []  # Here for demonstration purposes.
            temp_mapping.append((iread, iproj))

            kx = self.trajectory[iproj, iread, 1]
            ky = self.trajectory[iproj, iread, 0]

            ixmin = int((kx - self.kwidth) * self.grid_size + gridcenter)
            ixmax = int((kx + self.kwidth) * self.grid_size + gridcenter) + 1
            iymin = int((ky - self.kwidth) * self.grid_size + gridcenter)
            iymax = int((ky + self.kwidth) * self.grid_size + gridcenter) + 1

            for gcount1 in np.arange(ixmin, ixmax+1):
                dkx = (gcount1 - gridcenter) / self.grid_size - kx

                for gcount2 in np.arange(iymin, iymax+1):
                    dky = (gcount2 - gridcenter) / self.grid_size - ky
                    dk = np.sqrt(dkx ** 2 + dky ** 2)

                    if dk < self.kwidth:
                        fracind = dk / self.kwidth * (self.n_kernel_points - 1)
                        kernelind = int(fracind)
                        fracdk = fracind - kernelind

                        kern = (
                            self.kerneltable[kernelind] * (1 - fracdk) +
                            self.kerneltable[kernelind + 1] * fracdk
                            )

                        indx = gcount1
                        indy = gcount2

                        if gcount1 < 0:
                            indx += self.grid_size
                            indy = self.grid_size - indy

                        if gcount1 >= self.grid_size:
                            indx -= self.grid_size
                            indy = self.grid_size - indy

                        if gcount2 < 0:
                            indy += self.grid_size
                            indx = self.grid_size - indx

                        if gcount2 >= self.grid_size:
                            indy -= self.grid_size
                            indx = self.grid_size - indx

                        temp_mapping.append((indx, indy))  # Here for demonstration purposes

                        sg[:, indy, indx] += (
                            kern * kdat[
                                :,
                                iproj,
                                iread
                                ]
                            )

            grid_point_mapping.append(temp_mapping)  # Here for demonstration purposes

        if return_mapping:
            return sg, grid_point_mapping
        else:
            return sg

    def _invgrid_lut(self, sg):
        gridcenter = self.grid_size / 2

        s = np.zeros(
            (
                self.num_coils,
                self.num_proj,
                self.num_reads
                ),
            dtype=self.DTYPE
            )

        for iproj, ismpl in itertools.product(
                range(self.num_proj),
                range(self.num_reads)
                ):

            kx = self.trajectory[iproj, ismpl, 1]
            ky = self.trajectory[iproj, ismpl, 0]

            ixmin = int((kx - self.kwidth) * self.grid_size + gridcenter)
            ixmax = int((kx + self.kwidth) * self.grid_size + gridcenter) + 1
            iymin = int((ky - self.kwidth) * self.grid_size + gridcenter)
            iymax = int((ky + self.kwidth) * self.grid_size + gridcenter) + 1

            for gcount1 in np.arange(ixmin, ixmax+1):
                dkx = (gcount1 - gridcenter) / self.grid_size - kx

                for gcount2 in np.arange(iymin, iymax+1):
                    dky = (gcount2 - gridcenter) / self.grid_size - ky
                    dk = np.sqrt(dkx ** 2 + dky ** 2)
                    if dk < self.kwidth:

                        fracind = dk / self.kwidth * (self.n_kernel_points - 1)
                        kernelind = int(fracind)
                        fracdk = fracind - kernelind

                        kern = (
                            self.kerneltable[kernelind] * (1 - fracdk) +
                            self.kerneltable[kernelind + 1] * fracdk
                            )
                        indx = gcount1
                        indy = gcount2

                        if gcount1 < 0:
                            indx += self.grid_size
                            indy = self.grid_size - indy

                        if gcount1 >= self.grid_size:
                            indx -= self.grid_size
                            indy = self.grid_size - indy

                        if gcount2 < 0:
                            indy += self.grid_size
                            indx = self.grid_size - indx

                        if gcount2 >= self.grid_size:
                            indy -= self.grid_size
                            indx = self.grid_size - indx

                        s[:, iproj, ismpl] += \
                            kern*sg[:, indy, indx]
        return s * self.dens_comp

    def _generate_gridding_matrix(self):
            gridcenter = self.grid_size / 2
    
            rowind = []
            colind = []
            value = []
            for iproj, ismpl in itertools.product(
                    range(self.num_proj),
                    range(self.num_reads)
                    ):
    
                kx = self.trajectory[iproj, ismpl, 1]
                ky = self.trajectory[iproj, ismpl, 0]
    
                ixmin = int((kx - self.kwidth) * self.grid_size + gridcenter)
                ixmax = int((kx + self.kwidth) * self.grid_size + gridcenter) + 1
                iymin = int((ky - self.kwidth) * self.grid_size + gridcenter)
                iymax = int((ky + self.kwidth) * self.grid_size + gridcenter) + 1

                for gcount1 in np.arange(ixmin, ixmax+1):
                    dkx = (gcount1 - gridcenter) / self.grid_size - kx
                    for gcount2 in np.arange(iymin, iymax+1):
                        dky = (gcount2 - gridcenter) / self.grid_size - ky
                        dk = np.sqrt(dkx ** 2 + dky ** 2)
                        if dk < self.kwidth:
                            fracind = dk / self.kwidth * (self.n_kernel_points - 1)
                            kernelind = int(fracind)
                            fracdk = fracind - kernelind
                            kern = (
                                self.kerneltable[kernelind] * (1 - fracdk) +
                                self.kerneltable[kernelind + 1] * fracdk
                                )
                            indx = gcount1
                            indy = gcount2
    
                            if gcount1 < 0:
                                indx += self.grid_size
                                indy = self.grid_size - indy
                            if gcount1 >= self.grid_size:
                                indx -= self.grid_size
                                indy = self.grid_size - indy
                            if gcount2 < 0:
                                indy += self.grid_size
                                indx = self.grid_size - indx
                            if gcount2 >= self.grid_size:
                                indy -= self.grid_size
                                indx = self.grid_size - indx
                                
                            rowind.append(iproj*self.num_reads+ismpl)
                            colind.append(indy*self.grid_size+indx)
                            value.append(kern)

          
            gridmat = scipy.sparse.coo_matrix(
                (value, (rowind, colind)), 
                shape=(self.num_proj*self.num_reads,
                       self.grid_size**2))
            gridmat = gridmat.tocsc()
            return gridmat               


class MRIImagingModel(Operator):
    """
    The MRI imaging model including Coils.

    Linear Operator including coil sensitivities and fourier transform to
    transverse between k-space and image space.

    Attributes
    ----------
        coils (Numpy.Array):
            The complex coil sensitivity profiles
        conj_coils (Numpy.Array):
            The precomputed complex conjugate coil sensitivity profiles
        NUFFT (linop.NUFFT):
            The NUFFT object to perform gridding.
    """

    def __init__(
            self,
            par,
            trajectory
            ):
        """
        NUFFT object constructor.

        Args
        ----
            par (dict):
                A python dict containing the necessary information to
                setup the object. Needs to contain the number of slices
                (num_slc),
                number of scans (num_scans), image dimensions (dimX, dimY),
                number of coils (num_coils), sampling pos (num_reads)
                and read outs (num_proj) and the complex coil sensitivities
                (coils).
                The above is just not true. The __init__ statement says so.
                We use the keys 'Data' and 'FFT' here all of a sudden.
                Previously par contained only the content of 'Data'. This is very confusing.
            trajectory (numpy.array):
                Trajectory information for kx/ky/kz points.
        """
        super().__init__(par["Data"])

        self.NUFFT = NUFFT(
            data_par=par["Data"],
            fft_par=par["FFT"],
            trajectory=trajectory
            )
        self.coils = par["Data"]["coils"]
        self.conj_coils = np.conj(self.coils)

    def adjoint(self, inp):
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
        return np.sum(self.NUFFT.adjoint(inp) * self.conj_coils, 0)

    def forward(self, inp):
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
        return self.NUFFT.forward(inp * self.coils)
