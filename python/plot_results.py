#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:45:18 2018

@author: omaier
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
import h5py
import scipy.io as sio
import os

rc('text', usetex=True)
plt.close('all')


if os.path.exists('.'+os.sep+'output'+os.sep+'python'+os.sep+'brain'):
    cwd = os.getcwd()
    outdir = '.'+os.sep+'output'+os.sep+'python'+os.sep+'brain'
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

    Delta = []
    for j in range(1, num_recon):
        Delta.append(np.linalg.norm(data[j]-ref, axis=(-2, -1))**2 /
                    np.linalg.norm(ref)**2)
    Delta = np.array(Delta)

    ### Create Figure directory
    if not os.path.exists('.'+os.sep+'figures'+os.sep+'python'):
        os.makedirs('.'+os.sep+'figures'+os.sep+'python')


    # Brain ###################################################################
    plt.ion()
    figure = plt.figure(figsize=(6, 5))
    figure.tight_layout()
    plt.xlabel('Iterations')
    plt.ylabel('Log$_{10}$ $\delta$')
    plt.title("Brain Reconstruction $\delta$ criterion")
    ax_res = []
    labels = ["Acc 1", "Acc 2", "Acc 3", "Acc 4"]
    linestyle = ["-", ":", "-.", "--"]
    for j in range(num_recon):
        ax_res.append(plt.plot(np.log10(res[j]),
                              label=labels[j], linestyle=linestyle[j]))
    plt.legend()
    plt.savefig(
        '.'+os.sep+'figures'+os.sep+'python'
        +os.sep+'Conv_rate_small_delta.png')

    plt.ion()
    figure = plt.figure(figsize=(6, 5))
    figure.tight_layout()
    plt.xlabel('Iterations')
    plt.ylabel('Log$_{10}$ $\Delta$')
    plt.title("Brain Reconstruction $\Delta$ criterion")
    ax_res = []
    labels = ["Acc 2", "Acc 3", "Acc 4"]
    linestyle = [":", "-.", "--"]
    for j in range(num_recon-1):
        ax_res.append(plt.plot(np.log10(Delta[j]),
                              label=labels[j], linestyle=linestyle[j]))
    plt.legend()
    plt.savefig(
        '.'+os.sep+'figures'+os.sep+'python'
        +os.sep+'Conv_rate_big_delta.png',dpi=300)

    plt.ion()
    figure = plt.figure(figsize=(5, 6))
    figure.subplots_adjust(hspace=0.1, wspace=0.1)
    gs = gridspec.GridSpec(
                4, 3)
    gs.tight_layout(figure)
    figure.patch.set_facecolor('w')
    ax = []
    for grid in gs:
        ax.append(plt.subplot(grid))
        ax[-1].grid(False)
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])

    for j in range(data.shape[0]):
        ax[3*j].imshow(np.abs(coil_img[j][0]), cmap='gray')
        ax[3*j+1].imshow(np.abs(data[j][1]), cmap='gray')
        ax[3*j+2].imshow(np.abs(data[j][-1]), cmap='gray')
        ax[3*j].set_ylabel("Acc " + str(j+1), rotation=0, labelpad=20)
        ax[3*j+1].text(
          data[j].shape[-1]-20, data[j].shape[-1]-5, "1", color="w")
        ax[3*j+2].text(data[j].shape[-1]-50, data[j].shape[-1]-5,
                      str(data[j].shape[0]-1), color="w")
        if j == 0:
            ax[3*j].set_title("Single coil")
            ax[3*j+1].set_title("Initial")
            ax[3*j+2].set_title("Final")
    plt.savefig(
        '.'+os.sep+'figures'+os.sep+'python'
        +os.sep+'Comparison_Reconstruction_Brain.png')



# Heart #######################################################################
if os.path.exists('.'+os.sep+'output'+os.sep+'python'+os.sep+'heart'):
    cwd = os.getcwd()
    outdir = '.'+os.sep+'output'+os.sep+'python'+os.sep+'heart'
    os.chdir(outdir)
    files = os.listdir()
    files.sort()
    data = []
    res = []
    for file in files:
        tmp_file = h5py.File(file, 'r')
        res_name = []
        for name in tmp_file.items():
            res_name.append(name[0])
        data.append(np.squeeze(tmp_file[
          res_name[res_name.index('CG_reco')]][()]))
        res.append(tmp_file.attrs["residuals"])
        tmp_file.close()
    os.chdir(cwd)
    data = np.squeeze(np.array(data))
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

    plt.ion()
    figure = plt.figure(figsize=(8, 4))
    figure.subplots_adjust(hspace=0, wspace=0.05)
    gs = gridspec.GridSpec(
                1, 4)
    gs.tight_layout(figure)
    figure.patch.set_facecolor('w')
    ax = []
    for grid in gs:
        ax.append(plt.subplot(grid))
        ax[-1].grid(False)
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
    labels = ["55", "33", "22", "11"]
    for j in range(num_recon):
        ax[j].imshow(np.abs(data[j][-1]), cmap='gray')
        ax[j].text(
          data[j].shape[-1]-25, data[j].shape[-1]-5, labels[j], color="w")
    plt.savefig(
        '.'+os.sep+'figures'+os.sep+'python'+os.sep+'Heart.png', dpi=300)