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
import os
import h5py
import argparse
import configparser
from rrsg_cgreco._helper_fun.density_compensation \
    import get_density_from_gridding 
from rrsg_cgreco._helper_fun.est_coils import estimate_coil_sensitivities
import rrsg_cgreco.linop as linop
import rrsg_cgreco.solver as solver

DTYPE = np.complex64
DTYPE_real = np.float32


def _get_args(
      configfile='.'+os.sep+'python'+os.sep+'default', 
      pathtofile='.'+os.sep+'data'+os.sep+'rawdata_brain_radial_96proj_12ch.h5',
      undersampling_factor=1
      ):
    """
    Parse command line arguments.

    Args
    ----
        config (string):
            Name of config file to use (default).
            The file is assumed to be in the same folder where the script
            is run. If not specified, use default parameters.
        do_inscale (bool):
            Wether to perform intensity scaling. Defaults to True.
        do_denscor (bool):
            Switch to choose between reconstruction with (True) or
            without (False) density compensation. Defaults to True.
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
    Run the CG reco of radial data.

    Args
    ----
        config (string):
            Name of config file to use (default).
            The file is assumed to be in the same folder where the script
            is run. If not specified, use default parameters.

        inscale (bool):
            Wether to perform intensity scaling. Defaults to True.

        denscor (bool):
            Switch to choose between reconstruction with (True) or
            without (False) density compensation. Defaults to True.

        data (string):
            Full qualified path to the h5 data file.

        undersampling_factor (int):
             Desired acceleration compared to the number of
             spokes provided in data.
             E.g. 1 uses all available spokes 2 every 2nd.

        overgridding_factor (string):
            Ratio between Cartesian cropped grid and full regridded
            k-space grid.
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
        path (string):
            Full qualified path to the .h5 data file.
        undersampling_factor (int):
            Desired acceleration compared to the number of
            spokes provided in data.
            E.g. 1 uses all available spokes 2 every 2nd.
        par (dict):
            Dictionary for storing data and parameters.
    Retruns
    -------
        rawdata (np.complex64):
            The rawdata array
        trajectory (np.complex64):
            The k-space trajectory
    Raises
    ------
         ValueError:
             If no data file is specified
    """
    if not os.path.isfile(pathtofile):
        raise ValueError("Given path is not an existing file.")

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

    # Squeeze dummy dimension and transpose to C-style ordering.
    rawdata = np.squeeze(rawdata.T)

    # Normalize trajectory to the range of (-1/2)/(1/2)
    norm_trajectory = 2 * np.max(np.abs(trajectory))

    trajectory = (
      np.require(
        (trajectory / norm_trajectory).T,
        requirements='C'
        )
      )

    # Check if rawdata and trajectory dimensions match
    assert trajectory.shape[:-1] == rawdata.shape[-2:], \
        "Rawdata and trajectory should have the same number "\
        "of read/projection pairs."

    return rawdata, trajectory, noise_scan


def setup_parameter_dict(
      configfile,
      rawdata,
      trajectory
      ):
    """
    Parameter dict generation.

    Args
    ----
        rawdata (np.complex64):
            The raw k-space data

    Returns
    -------
        par (dict):
            A dictionary storing reconstruction related parameters like
            number of coils and image dimension in 2D.
    """
    # Create empty dict
    parameter = {}
    config = configparser.ConfigParser()
    if configfile.split('.')[-1] == "txt":
        pass
    else:
        configfile = configfile+'.txt'
    config.read(configfile)
    for sectionkey in config.sections():
        parameter[sectionkey] = {}
        for valuekey in config[sectionkey].keys():
            if "do_" in valuekey:
                try:  
                    parameter[sectionkey][valuekey] = config.getboolean(
                        sectionkey, 
                        valuekey)  
                except:
                    parameter[sectionkey][valuekey] = config.get(
                        sectionkey, 
                        valuekey)
            else:
                try:
                    parameter[sectionkey][valuekey] = config.getint(
                        sectionkey, 
                        valuekey)
                except:
                    try:
                        parameter[sectionkey][valuekey] = config.getfloat(
                            sectionkey, 
                            valuekey)
                    except:
                        parameter[sectionkey][valuekey] = config.get(
                            sectionkey, 
                            valuekey)
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
    parameter["Data"]["image_dim"] = int(
        num_reads/parameter["Data"]["overgridfactor"]
        )
    parameter["Data"]["num_reads"] = num_reads
    parameter["Data"]["num_proj"] = n_spokes
    
    # Calculate density compensation for non-cartesian data.
    if parameter["Data"]["do_density_correction"]:
        FFT = linop.NUFFT(data_par=parameter["Data"], 
                          fft_par=parameter["FFT"],
                          trajectory=trajectory)
        parameter["FFT"]["gridding_matrix"] = FFT.gridding_mat
        parameter["FFT"]["dens_cor"] = np.sqrt(
            get_density_from_gridding(
                parameter["Data"], 
                parameter["FFT"]["gridding_matrix"]
                )
            )
    else:
        parameter["FFT"]["dens_cor"] = np.ones(
            trajectory.shape[:-1],
            dtype=parameter["Data"]["DTYPE_real"]
            )
    return parameter


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
      args (ArgumentParser):
         Console arguments passed to the script.
    """
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


def _decor_noise(data, noise, par):
    if noise is None:
        return data
    else:
        cov = np.cov(noise)
        L = np.linalg.cholesky(cov)
        invL = np.linalg.inv(L)
        data = np.reshape(data, (par["num_coils"], -1))
        data = invL@data
        data = np.reshape(data,
                          (par["num_coils"],
                           par["num_proj"],
                           par["num_reads"]))
        return data


def _run_reco(args):
    # Read input data
    kspace_data, trajectory, noise = read_data(
        pathtofile=args.pathtofile, 
        undersampling_factor=args.undersampling_factor
        )
    # Setup parameters
    parameter = setup_parameter_dict(
        args.configfile,
        rawdata=kspace_data, 
        trajectory=trajectory)
    # Decorrelate Coil channels if noise scan is present
    kspace_data = _decor_noise(
        kspace_data, 
        noise, 
        parameter["Data"])
    # Get coil sensitivities in the parameter dict
    estimate_coil_sensitivities(
        kspace_data, 
        trajectory, 
        parameter)
    # Get operator
    MRImagingOperator = linop.MRIImagingModel(parameter, trajectory)
    cgs = solver.CGReco(
        data_par=parameter["Data"],
        optimizer_par=parameter["Optimizer"])
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
