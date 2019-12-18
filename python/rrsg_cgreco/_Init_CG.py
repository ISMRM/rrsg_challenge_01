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
from rrsg_cgreco._helper_fun import goldcomp as goldcomp

DTYPE = np.complex64
DTYPE_real = np.float32


def run(config='default', inscale=True, denscor=True,
        data='rawdata_brain_radial_96proj_12ch.h5', acc=1,
        ogf='1.706'):
    '''
    Function to run the CG reco of radial data.
    Args:
      config (string):
         Name of config file to use (default).
         The file is assumed to be in the same folder where the script is run.
         If not specified, use default parameters.
      inscale (bool):
         Wether to perform intensity scaling. Defaults to True.
      denscor (bool):
         Switch to choose between reconstruction with (True) or without (False)
         density compensation. Defaults to True.
      data (string):
         Full qualified path to the h5 data file.
      acc (int):
         Desired acceleration compared to the number of
         spokes provided in data.
         E.g. 1 uses all available spokes 2 every 2nd.
      ogf (string):
         Ratio between Cartesian cropped grid and full regridded k-space grid.
    '''
    parser = argparse.ArgumentParser(description='CG Sense Reconstruction')
    parser.add_argument(
      '--config', default=config, dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    parser.add_argument(
      '--inscale', default=inscale, type=int, dest='inscale',
      help='Perform Intensity Scaling.')
    parser.add_argument(
      '--denscor', default=denscor, type=int, dest='denscor',
      help='Perform density correction.')
    parser.add_argument(
      '--data', default=data, dest='data',
      help='Path to the h5 data file.')
    parser.add_argument(
      '--acc', default=acc, type=int, dest='acc',
      help='Desired acceleration factor.')
    parser.add_argument(
      '--ogf', default=ogf, type=str, dest='ogf',
      help='Overgridfactor. 1.706 for Brain, 1+1/3 for heart data.')
    args = parser.parse_args()
    _run_reco(args)


def _prepareData(path, acc):
    '''
    Reading in h5 data from file. And apply undersampling if specified.
    It is assumed that the data is saved as complex valued entry named
    "rawdata". The corresponding measurement trajectory is also saved as
    complex valued entry named "trajectory"
    Args:
      path (string):
         Full qualified path to the h5 data file.
      acc (int):
         Desired acceleration compared to the number of
         spokes provided in data.
         E.g. 1 uses all available spokes 2 every 2nd.
      par (dict):
         Dictionary for storing data and parameters.
   Retruns:
     rawdata (np.complex64):
       The rawdata array
     trajectory (np.complex64):
       The k-space trajectory
   Raises:
       ValueError:
         If no data file is specified
    '''

    if args.data == '':
        raise ValueError("No data file specified")

    name = os.path.normpath(path)
    h5_dataset = h5py.File(name, 'r')
    h5_dataset_rawdata_name = 'rawdata'
    h5_dataset_trajectory_name = 'trajectory'

    if "heart" in args.data:
        if args.acc == 2:
            R = 33
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
                :, :, :33]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
                :, :, :33, :]
        elif args.acc == 3:
            R = 22
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
                :, :, :22]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
                :, :, :22, :]
        elif args.acc == 4:
            R = 11
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
                :, :, :11]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
                :, :, :11, :]
        else:
            R = 55
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[...]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[...]
    else:
        R = args.acc
        trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
            :, :, ::R]
        rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
            :, :, ::R, :]

    # Squeeze dummy dimension and transpose to C-style ordering.
    rawdata = np.squeeze(rawdata.T)

    # Norm Trajectory to the range of (-1/2)/(1/2)
    trajectory = np.require((
        trajectory[0]/(2*np.max(trajectory[0])) +
        1j*trajectory[1]/(2*np.max(trajectory[0]))).T,
                   requirements='C')

    # Close file after everything was read
    h5_dataset.close()
    return (rawdata, trajectory)


def _setupParamterDict(rawdata, ogf):
    '''
    Setup the parameter dict.

    Args:
      rawdata (np.complex64):
        The raw k-space data
      ogf (string):
         Ratio between Cartesian cropped grid and full regridded k-space grid.
    Returns:
      par (dict):
        A dictionary storing reconstruction related parameters like number of
        coils and image dimension in 2D.
    '''
    # Create empty dict
    par = {}
    [nCh, nSpokes, nFE] = rawdata.shape

    par["ogf"] = float(eval(ogf))
    dimX, dimY = [int(nFE/par["ogf"]), int(nFE/par["ogf"])]

    # Calculate density compensation for radial data.
    #############
    # This needs to be adjusted for spirals!!!!!
    #############
    par["dcf"] = np.sqrt(np.array(goldcomp.cmp(
                     par["traj"]), dtype=DTYPE_real)).astype(DTYPE_real)
    par["dcf"] = np.require(np.abs(par["dcf"]),
                            DTYPE_real, requirements='C')

    par["NC"] = nCh
    par["dimY"] = dimY
    par["dimX"] = dimX
    par["nFE"] = nFE
    par["Nproj"] = nSpokes

    return par


def _saveToFile(result, args):
    '''
    Save the reconstruction result to a h5 file.

    Args:
      result (np.complex64):
        The reconstructed complex images to save.
      args (ArgumentParser):
         Console arguments passed to the script.
    '''
    outdir = ""
    if "heart" in args.data:
        outdir += "/heart"
    elif "brain" in args.data:
        outdir += "/brain"
    if not os.path.exists('./output'):
        os.makedirs('output')
    if not os.path.exists('./output' + outdir):
        os.makedirs("./output" + outdir)
    cwd = os.getcwd()
    os.chdir("./output" + outdir)
    f = h5py.File("CG_reco_inscale_" + str(args.inscale) + "_denscor_"
                  + str(args.denscor) + "_reduction_" + str(args.acc) +
                  "_acc_" + str(args.acc), "w")
    f.create_dataset("CG_reco", result.shape,
                     dtype=DTYPE, data=result)
    f.flush()
    f.close()
    os.chdir(cwd)


def _run_reco(args):
###############################################################################
# Read Input data   ###########################################################
###############################################################################
    (rawdata, trajectory) = _prepareData(args.data, args.acc)

###############################################################################
# Setup parameters #############################################
###############################################################################
    par = _setupParamterDict(rawdata, trajectory, args.ogf)
###############################################################################
# Coil Sensitivity Estimation #################################################
###############################################################################


###############################################################################
# generate nFFT  ##############################################################
###############################################################################
###############################################################################
# Start Reco ##################################################################
###############################################################################
###############################################################################
# New .hdf5 save files ########################################################
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CG Sense Reconstruction')
    parser.add_argument(
      '--config', default='default', dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    parser.add_argument(
      '--inscale', default=1, type=int, dest='inscale',
      help='Perform Intensity Scaling.')
    parser.add_argument(
      '--denscor', default=1, type=int, dest='denscor',
      help='Perform density correction.')
    parser.add_argument(
      '--data', default='rawdata_brain_radial_96proj_12ch.h5', dest='data',
      help='Path to the h5 data file.')
    parser.add_argument(
      '--acc', default=1, type=int, dest='acc',
      help='Desired acceleration factor.')
    parser.add_argument(
      '--ogf', default="1.706", type=str, dest='ogf',
      help='Overgridfactor. 1.706 for Brain, 1+1/3 for heart data.')
    args = parser.parse_args()
    _run_reco(args)
