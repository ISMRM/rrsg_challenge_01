#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main skript to run CG SENSE."""
import numpy as np
import os
import h5py
import argparse
import configparser
from rrsg_cgreco._helper_fun.density_compensation \
    import get_density_from_gridding
from rrsg_cgreco._helper_fun.est_coils import estimate_coil_sensitivities
import rrsg_cgreco.linop as linop
import rrsg_cgreco.solver as solver
import errno

DTYPE = np.complex64
DTYPE_real = np.float32


def _get_args(
      configfile='.'+os.sep+'python'+os.sep+'default',
      pathtofile=(
          '.'+os.sep+'data'+os.sep+'rawdata_brain_radial_96proj_12ch.h5'),
      undersampling_factor=1
      ):
    """
    Parse command line arguments.

    Args
    ----
        configfile (string):
            Path to the config file to use (default).
            The file is assumed to be in the same folder where the script
            is run. If not specified, use default parameters.
        pathtofile (string):
            Full qualified path to the h5 data file.

        undersampling_factor (int):
            Desired undersampling compared to the number of
            spokes provided in data.
            E.g. 1 uses all available spokes 2 every 2nd.

    Returns
    -------
        The parsed arguments as argparse object
    """
    parser = argparse.ArgumentParser(description='CG Sense Reconstruction')
    parser.add_argument(
        '--config', default=configfile, dest='configfile',
        help='Name of config file to use (assumed to be in the same folder). '
             'If not specified, use default parameters.'
             )
    parser.add_argument(
        '--datafile', default=pathtofile, dest='pathtofile',
        help='Path to the h5 data file.'
        )
    parser.add_argument(
        '--acc', default=undersampling_factor, type=int,
        dest='undersampling_factor',
        help='Desired undersampling factor.'
        )
    args = parser.parse_args()
    return args


def run(
      configfile='.'+os.sep+'python'+os.sep+'default',
      datafile='.'+os.sep+'data'+os.sep+'rawdata_brain_radial_96proj_12ch.h5',
      undersampling_factor=1,
      ):
    """
    Run the CG reco of radial or spiral data.

    Args
    ----
        configfile (string):
            Path to the config file to use (default).
            The file is assumed to be in the same folder where the script
            is run. If not specified, use default parameters.
        pathtofile (string):
            Full qualified path to the h5 data file.

        undersampling_factor (int):
            Desired undersampling compared to the number of
            spokes provided in data.
            E.g. 1 uses all available spokes 2 every 2nd.
    """
    args = _get_args(
        configfile,
        datafile,
        undersampling_factor
        )
    _run_reco(args)


def read_data(
      pathtofile,
      undersampling_factor,
      data_rawdata_key='rawdata',
      data_trajectory_key='trajectory',
      noise_key='noise'
      ):
    """
    Handle data and possible undersampling.

    Reading in h5 data from the path variable.
    Apply undersampling if specified.
    It is assumed that the data is saved as complex
    valued entry named "rawdata".
    The corresponding measurement trajectory is also saved as complex valued
    entry named "trajectory"

    Args
    ----
        pathtofile (string):
            Full qualified path to the .h5 data file.
        undersampling_factor (int):
            Desired acceleration compared to the number of
            spokes provided in data.
            E.g. 1 uses all available spokes 2 every 2nd.
        data_rawdata_key (string):
            Name of the data array in the .h5 file. defaults to "rawdata"
        data_trajectory_key (string):
            Name of the trajectory array in the .h5 file. defaults to
            "trajectory"
        noise_key (string):
            Name of the noise reference array in the .h5 file.
            defaults to "noise"

    Retruns
    -------
        rawdata (np.complex64):
            The rawdata array
        trajectory (np.complex64):
            The k-space trajectory
        noise_scan (np.complex64):
            The noise reference scan
        data_par (struct):
            Optional data parameters derived from the trajectory
        Coils (np.complex64):
            If available, read in complex coil senstivity profiles.
            Defaults to None.

    Raises
    ------
         ValueError:
             If no data file is specified
    """
    if not os.path.isfile(pathtofile):
        err = FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            pathtofile)
        raise FileNotFoundError(
            "Given path does not point to an existing file:\n{0}".format(
                err))

    name = os.path.normpath(pathtofile)
    with h5py.File(name, 'r') as h5_dataset:
        if "heart" in name:
            if undersampling_factor == 2:
                trajectory = h5_dataset[data_trajectory_key][:, :, :33]
                rawdata = h5_dataset[data_rawdata_key][:, :, :33, :]
            elif undersampling_factor == 3:
                trajectory = h5_dataset[data_trajectory_key][:, :, :22]
                rawdata = h5_dataset[data_rawdata_key][:, :, :22, :]
            elif undersampling_factor == 4:
                trajectory = h5_dataset[data_trajectory_key][:, :, :11]
                rawdata = h5_dataset[data_rawdata_key][:, :, :11, :]
            else:
                trajectory = h5_dataset[data_trajectory_key][...]
                rawdata = h5_dataset[data_rawdata_key][...]
        else:
            trajectory = h5_dataset[data_trajectory_key][
                :, :, ::undersampling_factor]
            rawdata = h5_dataset[data_rawdata_key][
                :, :, ::undersampling_factor, :]
        if noise_key in h5_dataset.keys():
            noise_scan = h5_dataset[noise_key][()]
        else:
            noise_scan = None
        if "Coils" in h5_dataset.keys():
            Coils = h5_dataset["Coils"][()]
        else:
            Coils = None
        if "mask" in h5_dataset.keys():
            mask = h5_dataset["mask"][()]
        else:
            mask = 1

    # Squeeze dummy dimension and transpose to C-style ordering.
    rawdata = np.squeeze(rawdata.T)

    # Derive image dimension and overgridfactor from trajectory
    center_out_scale = 1
    if np.allclose(np.round(trajectory[:, 0, :]), 0):
        print("k-space seems measured center out.")
        center_out_scale = 2

    image_dim = int(
        center_out_scale *
        np.max(
            np.linalg.norm(
                trajectory[:, -1, :] - trajectory[:, 0, :], axis=0
                )
            )
        )

    if np.mod(image_dim, 2):
        print("Uneven image dimension: "+str(image_dim)+", increasing by 1.")
        image_dim += 1

    overgrid_factor_a = 1/np.linalg.norm(
        trajectory[:, -2, 0]-trajectory[:, -1, 0])
    overgrid_factor_b = 1/np.linalg.norm(
        trajectory[:, 0, 0]-trajectory[:, 1, 0])

    data_par = {}
    data_par["overgridfactor"] = np.min((overgrid_factor_a,
                                         overgrid_factor_b))
    data_par["image_dimension"] = image_dim
    data_par["mask"] = mask

    # Transpose trajectory to projections/reads/position order
    trajectory = (
      np.require(
        (trajectory).T,
        requirements='C'
        )
      )
    # Check if rawdata and trajectory dimensions match
    assert trajectory.shape[:-1] == rawdata.shape[-2:], \
        "Rawdata and trajectory should have the same number "\
        "of read/projection pairs."

    if image_dim < 10:
        image_dim = None

    return rawdata, trajectory, noise_scan, data_par, Coils


def setup_parameter_dict(
      configfile,
      rawdata,
      trajectory,
      data_par=None
      ):
    """
    Parameter dict generation.

    This function reads in the parameters given in the configfile as well
    as some general information about the data and trajectory such as
    image size.

    Args
    ----
        configfile (str):
            path to configuration file
        rawdata (np.complex64):
            The raw k-space data
        trajectory (np.array):
            The associated trajectory data
        data_par (dcit):
            The image grid dimension and
            overgrid factor, dervied from the trajectory

    Returns
    -------
        parameter (dict):
            A dictionary storing reconstruction related parameters like
            number of coils and image dimension in 2D.
    """
    # Create empty dict
    parameter = {}
    config = configparser.ConfigParser()
    ext = os.path.splitext(configfile)[-1]
    if ext != ".txt":
        configfile = configfile + '.txt'
    try:
        config.read_file(open(configfile))
        config.read(configfile)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            "Given Path doesn't point to an existing config file:\n{0}".format(
                err))

    for section_key in config.sections():
        parameter[section_key] = {}
        for value_key in config[section_key].keys():
            if "do_" in value_key:
                try:
                    parameter[section_key][value_key] = config.getboolean(
                        section_key,
                        value_key)
                except ValueError:
                    parameter[section_key][value_key] = config.get(
                        section_key,
                        value_key)
            else:
                try:
                    parameter[section_key][value_key] = config.getint(
                        section_key,
                        value_key)
                except ValueError:
                    try:
                        parameter[section_key][value_key] = config.getfloat(
                            section_key,
                            value_key)
                    except ValueError:
                        parameter[section_key][value_key] = config.get(
                            section_key,
                            value_key)
    parameter["Data"] = {**parameter["Data"], **data_par}
    if parameter["Data"]["precision"].lower() == "single":
        parameter["Data"]["DTYPE"] = np.complex64
        parameter["Data"]["DTYPE_real"] = np.float32
    elif parameter["Data"]["precision"].lower() == "double":
        parameter["Data"]["DTYPE"] = np.complex128
        parameter["Data"]["DTYPE_real"] = np.float64
    else:
        raise ValueError("precision needs to be set to single or double.")

    [n_ch, n_spokes, num_reads] = rawdata.shape

    parameter["Data"]["num_coils"] = n_ch
    parameter["Data"]["num_reads"] = num_reads
    parameter["Data"]["num_proj"] = n_spokes
    parameter["Data"]["grid_size"] = int(np.ceil(
        parameter["Data"]["image_dimension"] *
        parameter["Data"]["overgridfactor"]))

    # Calculate density compensation for non-cartesian data.
    if parameter["Data"]["do_density_correction"]:
        print("Estimating gridding density...")
        compute_density_compensation(parameter, trajectory)

    else:
        parameter["FFT"]["dens_cor"] = np.ones(
            trajectory.shape[:-1],
            dtype=parameter["Data"]["DTYPE_real"]
            )
    return parameter


def compute_density_compensation(parameter, trajectory):
    """
    Compensate for non uniform sampling density.

    This function computes the sampling density via gridding of ones and
    the correct intensity normalization of the NUFFT operator.

    Args
    ----
        parameter (dict):
            A dictionary storing reconstruction related parameters like
           number of coils and image dimension in 2D.
        trajectory (np.array):
            The associated trajectory data
    """
    # First setup a NUFFT with the given trajectroy
    FFT = linop.NUFFT(par=parameter,
                      trajectory=trajectory)
    # Extrakt the gridding matrix
    parameter["FFT"]["gridding_matrix"] = FFT.gridding_mat
    # Grid a k-space of all ones to get an estimated density
    # and use it as density compensation
    parameter["FFT"]["dens_cor"] = np.sqrt(
        get_density_from_gridding(
            parameter["Data"],
            parameter["FFT"]["gridding_matrix"]
            )
        )


def save_to_file(
      result,
      residuals,
      single_coil_images,
      data_par,
      args
      ):
    """
    Save the reconstruction result to a h5 file.

    Args
    ----
      result (np.complex64):
        The reconstructed complex images to save.
      residuals (list):
        List of residual values of the CG algorithm.
      single_coil_images (np.complex64):
        The complex single coil image to save.
      data_par (dict):
        The data parameter dict.
      args (ArgumentParser):
         Console arguments passed to the script.
    """
    print("Saving results...")
    outdir = ""
    if "heart" in args.pathtofile:
        outdir += os.sep+'heart'
    elif "brain" in args.pathtofile:
        outdir += os.sep+'brain'
    if not os.path.exists('.'+os.sep+'output'+os.sep+'python'):
        os.makedirs('output')
    if not os.path.exists('.'+os.sep+'output'+os.sep+'python' + outdir):
        os.makedirs('.'+os.sep+'output'+os.sep+'python' + outdir)
    cwd = os.getcwd()
    os.chdir('.'+os.sep+'output'+os.sep+'python' + outdir)
    f = h5py.File(
        "CG_reco_inscale_" + str(data_par["do_intensity_scale"]) + "_denscor_"
        + str(data_par["do_density_correction"]) +
        "_reduction_" + str(args.undersampling_factor)
        + ".h5",
        "w"
        )
    f.create_dataset(
        "CG_reco",
        result.shape,
        dtype=DTYPE,
        data=result
        )
    f.create_dataset(
        "Coil_images",
        single_coil_images.shape,
        dtype=DTYPE,
        data=single_coil_images
        )
    f.attrs["residuals"] = residuals
    f.flush()
    f.close()
    os.chdir(cwd)


def _decor_noise(data, noise, par, coils=None):
    """
    Decorrelate the data with using a given noise covariance matrix.

    Perform prewithening with given noise data. If no data was aquired,
    the input data without modifications is returned.

    Args
    ----
      data (np.complex64):
        The complex imput data.
      noise (np.complex64):
        The complex noise covariance data.
      par (dict):
        The data parameter dict.
      coils (np.complex64):
        The optional complex coil sensitivity data.

    Returns
    -------
        data (np.complex64):
            The prewithened k-space data
          coils (np.complex64):
            The corresponding coil sensitivities, if provided.
    """
    if noise is None:
        return data, coils
    else:
        print("Performing noise decorrelation...")
        if not np.allclose(noise.shape, par["num_coils"]):
            cov = np.cov(np.reshape(noise, (par["num_coils"], -1)))
        else:
            cov = noise
        L = np.linalg.cholesky(cov)
        invL = np.linalg.inv(L)
        data = np.reshape(data, (par["num_coils"], -1))
        data = invL@data
        data = np.reshape(data,
                          (par["num_coils"],
                           par["num_proj"],
                           par["num_reads"]))
        if coils is not None:
            coilshape = coils.shape
            coils = np.reshape(coils, (par["num_coils"], -1))
            coils = invL@coils
            coils = np.reshape(coils,
                               coilshape)
        return data, coils


def _save_coil_(pathtofile, undersampling_factor, par):
    name = os.path.normpath(pathtofile)
    with h5py.File(name, 'r+') as h5_dataset:
        if (undersampling_factor != 1) and ("Coils" not in h5_dataset.keys()):
            raise ValueError(
                "Coils should be estimated without undersampling!")
        elif "Coils" in h5_dataset.keys():
            pass
        else:
            h5_dataset["Coils"] = par["coils"]
            h5_dataset["mask"] = par["mask"]


def _run_reco(args):
    # Read input data
    kspace_data, trajectory, noise, data_par, coils = read_data(
        pathtofile=args.pathtofile,
        undersampling_factor=args.undersampling_factor
        )
    # Setup parameters
    parameter = setup_parameter_dict(
        args.configfile,
        rawdata=kspace_data,
        trajectory=trajectory,
        data_par=data_par)

    # Decorrelate Coil channels if noise scan is present
    kspace_data, coils = _decor_noise(
        data=kspace_data,
        noise=noise,
        par=parameter["Data"],
        coils=coils)

    # Get coil sensitivities in the parameter dict
    estimate_coil_sensitivities(
        kspace_data,
        trajectory,
        parameter,
        coils=coils)
    # Save coils if not present
    _save_coil_(
        pathtofile=args.pathtofile,
        undersampling_factor=args.undersampling_factor,
        par=parameter["Data"]
        )

    # Get operator
    MRImagingOperator = linop.MRIImagingModel(parameter, trajectory)
    cgs = solver.CGReco(
        data_par=parameter["Data"],
        optimizer_par=parameter["Optimizer"])
    print("Starting reconstruction...")
    cgs.set_operator(MRImagingOperator)
    # Start reconstruction
    # Data needs to be multiplied with the sqrt of dense_cor to assure that
    # forward and adjoint application of the NUFFT is adjoint with each other.
    # dens_cor itself is saved in the par dict as the sqrt.
    recon_result, residuals = cgs.optimize(
        data=kspace_data * parameter["FFT"]["dens_cor"]
        )

    # Single Coil images after FFT
    single_coil_images = cgs.operator.NUFFT.adjoint(
        kspace_data * parameter["FFT"]["dens_cor"])

    # Store results
    save_to_file(recon_result,
                 residuals,
                 single_coil_images,
                 parameter["Data"],
                 args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CG Sense Reconstruction')
    parser.add_argument(
        '--config',
        default='.'+os.sep+'python'+os.sep+'default',
        dest='configfile',
        help='Path to the config file to use. '
             'If not specified, use default parameters.'
             )
    parser.add_argument(
        '--datafile',
        default='.'+os.sep+'data'+os.sep+'rawdata_brain_radial_96proj_12ch.h5',
        dest='pathtofile',
        help='Path to the .h5 data file.'
        )
    parser.add_argument(
        '--acc', default=1, type=int, dest='undersampling_factor',
        help='Desired acceleration factor.'
        )
    args = parser.parse_args()
    _run_reco(args)
