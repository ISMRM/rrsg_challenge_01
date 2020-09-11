#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:30:35 2020

@author: omaier
"""

import prepare_results
import numpy as np
import h5py
import os
import scipy.io as sio

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

from skimage.metrics import structural_similarity as ssim


rc('text', usetex=True)
plt.close('all')

def doPlot(data, prefix='', vmin=None, vmax=None, doabs=False):
    # Brain ###################################################################
    if doabs:
        def x(a):
            return np.abs(a)
    else:
        def x(a):
            return a

    for key, value in data.items():
        plt.ion()
        figure = plt.figure(figsize=(4.4, 6))
        figure.subplots_adjust(hspace=0, wspace=0)
        gs = gridspec.GridSpec(
                    4, 3)
        figure.patch.set_facecolor('w')
        ax = []
        for grid in gs:
            ax.append(plt.subplot(grid))
            ax[-1].grid(False)
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])

        for j in range(4):
            ax[3*j].imshow(x(value["brain"][j, 0]), cmap='gray',
                           vmin=vmin, vmax=vmax)
            ax[3*j+1].imshow(x(value["brain"][j, 1]), cmap='gray',
                             vmin=vmin, vmax=vmax)
            ax[3*j+2].imshow(x(value["brain"][j, 2]), cmap='gray',
                             vmin=vmin, vmax=vmax)
            ax[3*j].set_ylabel("Acc " + str(j+1), rotation=0, labelpad=20,
                               color='w')
            ax[3*j+1].text(
              value["brain"][j, 0].shape[-1]-20,
              value["brain"][j, 0].shape[-1]-5,
              "1", color="w")
            ax[3*j+2].text(
                value["brain"][j, 0].shape[-1]-50,
                value["brain"][j, 0].shape[-1]-5,
                "10", color='w')
            if j == 0:
                ax[3*j].set_title("Single coil", color='white')
                ax[3*j+1].set_title("Initial", color='white')
                ax[3*j+2].set_title("Final", color='white')
        plt.savefig("./figures/"+prefix+"Brain_"+key+".png",
                    dpi=300,
                    facecolor='black',
                    bbox_inches='tight',
                    pad_inches=0)
        plt.savefig("./figures/"+prefix+"Brain_"+key+".svg",
                    dpi=300,
                    facecolor='black',
                    bbox_inches='tight',
                    pad_inches=0)
    spokes = [55, 33, 22, 11]
    # Heart ###################################################################
    for key, value in data.items():
        plt.ion()
        figure = plt.figure(figsize=(8, 4))
        figure.subplots_adjust(hspace=0, wspace=0)
        gs = gridspec.GridSpec(
                    1, 4)
        # gs.tight_layout(figure)
        figure.patch.set_facecolor('w')
        ax = []
        for grid in gs:
            ax.append(plt.subplot(grid))
            ax[-1].grid(False)
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
        # labels = ["55", "33", "22", "11"]
        for j in range(4):
            ax[j].imshow(x(value["heart"][j]), cmap='gray',
                         vmin=vmin, vmax=vmax)
            ax[j].set_title(str(spokes[j])+" Spokes", color='white')
        plt.savefig("./figures/"+prefix+"Heart_"+key+".png",
                    dpi=300,
                    facecolor='black',
                    bbox_inches='tight',
                    pad_inches=0)
        plt.savefig("./figures/"+prefix+"Heart_"+key+".svg",
                    dpi=300,
                    facecolor='black',
                    bbox_inches='tight',
                    pad_inches=0)
    plt.close('all')


# Please check that the hardcoded folders match your folder structure

file = h5py.File(
    "/mnt/data/Code/CGSENSE_challenge_sub/data/rawdata_heart_radial_55proj_34ch.h5", 'r+')
heart_mask = np.flip(file["mask"][240:-240,240:-240].astype(bool), -2)
file.close()
file= h5py.File("/mnt/data/Code/CGSENSE_challenge_sub/data/rawdata_brain_radial_96proj_12ch.h5", 'r+')
brain_mask = np.flip(file["mask"][190:-190,190:-190].astype(bool), axis=(-1,-2))
file.close()


data_load = prepare_results.read()
data = prepare_results.sort(data_load)

diff = {}
ssimval = {}
nrmse = {}
for key_sub, data_sub in data.items():
    if key_sub != "Ref_python":
        if key_sub not in diff.keys():
            diff[key_sub] = {}
        for key_region, data_region in data_sub.items():
            if key_region not in diff[key_sub].keys():
                diff[key_sub][key_region] = {}
            dims = data[key_sub][key_region].shape[-2:]
            dims_ref = data["Ref_python"][key_region].shape[-2:]

            tmp_dat = np.abs(data[key_sub][key_region]/np.quantile(
                np.abs(data[key_sub][key_region]), 0.95, axis=(-2, -1)
                )[..., None, None])
            normed_ref = np.abs(data["Ref_python"][key_region]/np.quantile(
                np.abs(data["Ref_python"][key_region]), 0.95, axis=(-2, -1)
                )[..., None, None])
            if key_region == "brain":
                mask = brain_mask
            else:
                mask = heart_mask
            normed_ref[normed_ref == 0] = 1
            if dims > dims_ref:
                indy = slice(
                    int((dims[0]-dims_ref[0])/2), int((dims[0]+dims_ref[0])/2))
                indx = slice(
                    int((dims[1]-dims_ref[1])/2), int((dims[1]+dims_ref[1])/2))
                diff[key_sub][key_region] = (
                    tmp_dat[..., indy, indx]
                    - normed_ref) / normed_ref * 100 * mask
            elif dims == dims_ref:
                indy = slice(
                    0, dims_ref[0])
                indx = slice(
                    0, dims_ref[1])
                diff[key_sub][key_region] = (
                    tmp_dat[..., indy, indx]
                    - normed_ref) / normed_ref * 100 * mask

            else:
                indy = slice(
                    int((-dims[0]+dims_ref[0])/2),
                    int((dims[0]+dims_ref[0])/2))
                indx = slice(
                    int((-dims[1]+dims_ref[1])/2),
                    int((dims[1]+dims_ref[1])/2))
                diff[key_sub][key_region] = (
                    tmp_dat
                    - normed_ref[
                        ...,
                        indy,
                        indx]) / normed_ref[
                        ...,
                        indy,
                        indx] * 100 * mask[..., indy, indx]
            if key_sub == "Ref_matlab":
                ssimval[key_region] = np.zeros(
                    (data[key_sub][key_region].shape[:-2]))
                nrmse[key_region] = np.zeros(
                    (data[key_sub][key_region].shape[:-2]))
                for i in range(normed_ref.shape[0]):
                    if len(normed_ref.shape) > 3:
                        for j in range(normed_ref.shape[1]):
                            ssimval[key_region][i, j] = (
                                ssim(
                                     tmp_dat[i, j, indy, indx][mask],
                                     normed_ref[i, j][mask],
                                     data_range=(
                                         normed_ref[i, j][mask].max()
                                         - normed_ref[i, j][mask].min()),
                                     gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False))
                    else:
                        ssimval[key_region][i] = (
                            ssim(tmp_dat[i, indy, indx][mask],
                                 normed_ref[i][mask],
                                 data_range=(
                                     normed_ref[i][mask].max()
                                     - normed_ref[i][mask].min()),
                                 gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False))
                    norm = np.mean(normed_ref[..., mask],
                                   axis=(-1))
                    nrmse[key_region] = np.sqrt(1/np.prod(dims_ref)*np.sum(
                        ((tmp_dat[..., indy, indx][..., mask])
                         - normed_ref[..., mask])**2,
                        axis=(-1)))/norm

doPlot(data, doabs=True)
doPlot(diff, 'diff', -10, 10)



if os.path.exists('..'+os.sep+'output'+os.sep+'python'+os.sep+'cardiac'):
    matres = sio.loadmat('/mnt/data/Code/CGSENSE_challenge_sub/output/matlab/cardiac/result_KI.mat')
    init = matres["initial"]
    sc = matres["singleCoil"]
    final = matres["final"]
    cwd = os.getcwd()
    outdir = '..'+os.sep+'output'+os.sep+'python'+os.sep+'cardiac'
    os.chdir(outdir)
    files = os.listdir()
    files.sort()
    data = []
    coil_img = []
    res = []
    for file in files:
        tmp_file = h5py.File(file, 'r')
        res_name = []
        for name in tmp_file.items():
            res_name.append(name[0])
        data.append(np.squeeze(tmp_file[res_name[
          res_name.index('CG_reco')]][()]))
        coil_img.append(np.squeeze(tmp_file[res_name[
            res_name.index('Coil_images')]][()]))
        res.append(tmp_file.attrs["residuals"])
        tmp_file.close()
    os.chdir(cwd)
    data = np.squeeze(np.array(data))
    coil_img = np.array(coil_img)
    res = np.array(res)

    if len(data.shape) == 3:
        ref = data[-1]
        data = data[None,...]
        num_recon = 1
    else:
        ref = data[0][-1]
        num_recon = data.shape[0]

    ### Create Figure directory
    if not os.path.exists('.'+os.sep+'figures'+os.sep+'python'):
        os.makedirs('.'+os.sep+'figures'+os.sep+'python')


    # Brain ###################################################################
    plt.ion()
    figure = plt.figure(figsize=(5, 6))
    figure.subplots_adjust(hspace=0.1, wspace=0.1)
    gs = gridspec.GridSpec(
                2, 3)
    gs.tight_layout(figure)
    figure.patch.set_facecolor('w')
    ax = []
    for grid in gs:
        ax.append(plt.subplot(grid))
        ax[-1].grid(False)
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])

    labels = ["Python" , "Matlab"]

    for j in range(data.shape[0]):
        ax[3*j].imshow(np.abs(coil_img[j][0]), cmap='gray')
        ax[3*j+1].imshow(np.abs(data[j][1]), cmap='gray')
        ax[3*j+2].imshow(np.abs(data[j][-1]), cmap='gray')
        ax[3*j].set_ylabel(labels[j], rotation=0, labelpad=20)
        ax[3*j+1].text(
          data[j].shape[-1]-20, data[j].shape[-1]-5, "1", color="w")
        ax[3*j+2].text(data[j].shape[-1]-50, data[j].shape[-1]-5,
                      str(data[j].shape[0]-1), color="w")
        if j == 0:
            ax[3*j].set_title("Single coil")
            ax[3*j+1].set_title("Initial")
            ax[3*j+2].set_title("Final")
    j=1
    ax[3*j].imshow(np.abs(sc)/np.abs(sc).max(), cmap='gray')
    ax[3*j+1].imshow(np.abs(init)/np.abs(init).max(), cmap='gray')
    ax[3*j+2].imshow(np.abs(final)/np.abs(final).max(), cmap='gray')
    ax[3*j].set_ylabel("Matlab", rotation=0, labelpad=20)
    ax[3*j+1].text(
      data[0].shape[-1]-20, data[0].shape[-1]-5, "1", color="w")
    ax[3*j+2].text(data[0].shape[-1]-50, data[0].shape[-1]-5,
                  str(data[0].shape[0]-1), color="w")
    plt.savefig(
        '.'+os.sep+'figures'+os.sep+'python'
        +os.sep+'Comparison_Reconstruction_KI.svg')


if os.path.exists('..'+os.sep+'output'+os.sep+'python'+os.sep+'spiral'):
    matres = sio.loadmat('/mnt/data/Code/CGSENSE_challenge_sub/output/matlab/spiral/result_spiral.mat')
    init = matres["initial"]
    sc = matres["singleCoil"]
    final = matres["final"]
    cwd = os.getcwd()
    outdir = '..'+os.sep+'output'+os.sep+'python'+os.sep+'spiral'
    os.chdir(outdir)
    files = os.listdir()
    files.sort()
    data = []
    coil_img = []
    res = []
    for file in files:
        tmp_file = h5py.File(file, 'r')
        res_name = []
        for name in tmp_file.items():
            res_name.append(name[0])
        data.append(np.squeeze(tmp_file[res_name[
          res_name.index('CG_reco')]][()]))
        coil_img.append(np.squeeze(tmp_file[res_name[
            res_name.index('Coil_images')]][()]))
        res.append(tmp_file.attrs["residuals"])
        tmp_file.close()
    os.chdir(cwd)
    data = np.squeeze(np.array(data))
    coil_img = np.array(coil_img)
    res = np.array(res)

    if len(data.shape) == 3:
        ref = data[-1]
        data = data[None,...]
        num_recon = 1
    else:
        ref = data[0][-1]
        num_recon = data.shape[0]

    ### Create Figure directory
    if not os.path.exists('.'+os.sep+'figures'+os.sep+'python'):
        os.makedirs('.'+os.sep+'figures'+os.sep+'python')


    # Brain ###################################################################
    plt.ion()
    figure = plt.figure(figsize=(5, 6))
    figure.subplots_adjust(hspace=0.1, wspace=0.1)
    gs = gridspec.GridSpec(
                2, 3)
    gs.tight_layout(figure)
    figure.patch.set_facecolor('w')
    ax = []
    for grid in gs:
        ax.append(plt.subplot(grid))
        ax[-1].grid(False)
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])

    labels = ["Python" , "Matlab"]

    for j in range(data.shape[0]):
        ax[3*j].imshow(np.abs(coil_img[j][0]), cmap='gray')
        ax[3*j+1].imshow(np.abs(data[j][1]), cmap='gray')
        ax[3*j+2].imshow(np.abs(data[j][-1]), cmap='gray')
        ax[3*j].set_ylabel(labels[j], rotation=0, labelpad=20)
        ax[3*j+1].text(
          data[j].shape[-1]-20, data[j].shape[-1]-5, "1", color="w")
        ax[3*j+2].text(data[j].shape[-1]-50, data[j].shape[-1]-5,
                      str(data[j].shape[0]-1), color="w")
        if j == 0:
            ax[3*j].set_title("Single coil")
            ax[3*j+1].set_title("Initial")
            ax[3*j+2].set_title("Final")
    j=1
    ax[3*j].imshow(np.abs(sc)/np.abs(sc).max(), cmap='gray')
    ax[3*j+1].imshow(np.abs(init)/np.abs(init).max(), cmap='gray')
    ax[3*j+2].imshow(np.abs(final)/np.abs(final).max(), cmap='gray')
    ax[3*j].set_ylabel("Matlab", rotation=0, labelpad=20)
    ax[3*j+1].text(
      data[0].shape[-1]-20, data[0].shape[-1]-5, "1", color="w")
    ax[3*j+2].text(data[0].shape[-1]-50, data[0].shape[-1]-5,
                  str(data[0].shape[0]-1), color="w")
    plt.savefig(
        '.'+os.sep+'figures'+os.sep+'python'
        +os.sep+'Comparison_Reconstruction_Spiral.svg')
