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
import os

rc('text', usetex=True)
plt.close('all')

def run():
    cwd = os.getcwd()
    outdir = "./output/brain/"
    os.chdir(outdir)
    files = os.listdir()
    files.sort()
    data = []
    coil_img = []
    res = []
    for file in files:
        tmp_file = h5py.File(file)
        res_name = []
        for name in tmp_file.items():
            res_name.append(name[0])
        data.append(np.squeeze(tmp_file[res_name[
          res_name.index('CG_reco')]][()]))
        coil_img.append(np.squeeze(tmp_file[res_name[
            res_name.index('images_ifft_coils_')]][()]))
        res.append(tmp_file.attrs["res"])
        tmp_file.close()
    os.chdir(cwd)
    data = np.squeeze(np.array(data))
    res = np.array(res)

    ref = data[0][-1]

    Delta = []
    for j in range(1, data.shape[0]):
        Delta.append(np.linalg.norm(data[j]-ref, axis=(-2, -1))**2 /
                    np.linalg.norm(ref)**2)
    Delta = np.array(Delta)
    [iters, y, x] = data[0].shape

# Brain ########################################################################
    plt.ion()
    figure = plt.figure(figsize=(6, 5))
    figure.tight_layout()
    plt.xlabel('Iterations')
    plt.ylabel('Log$_{10}$ $\delta$')
    plt.title("Brain Reconstruction $\delta$ criterion")
    ax_res = []
    labels = ["Acc 1", "Acc 2", "Acc 3", "Acc 4"]
    linestyle = ["-", ":", "-.", "--"]
    for j in range(res.shape[0]):
        ax_res.append(plt.plot(np.log10(res[j]),
                              label=labels[j], linestyle=linestyle[j]))
    plt.legend()
    plt.savefig("./figures/Conv_rate_small_delta.png")

    plt.ion()
    figure = plt.figure(figsize=(6, 5))
    figure.tight_layout()
    plt.xlabel('Iterations')
    plt.ylabel('Log$_{10}$ $\Delta$')
    plt.title("Brain Reconstruction $\Delta$ criterion")
    ax_res = []
    labels = ["Acc 2", "Acc 3", "Acc 4"]
    linestyle = [":", "-.", "--"]
    for j in range(Delta.shape[0]):
        ax_res.append(plt.plot(np.log10(Delta[j]),
                              label=labels[j], linestyle=linestyle[j]))
    plt.legend()
    plt.savefig("./figures/Conv_rate_big_delta.png",dpi=300)

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
    plt.savefig("./figures/Comparison_Reconstruction_Brain.png")
# Heart ########################################################################
    cwd = os.getcwd()
    outdir = "./output/heart/"
    os.chdir(outdir)
    files = os.listdir()
    files.sort(reverse=True)
    data = []
    res = []
    for file in files:
        tmp_file = h5py.File(file)
        res_name = []
        for name in tmp_file.items():
            res_name.append(name[0])
        data.append(np.squeeze(tmp_file[
          res_name[res_name.index('CG_reco')]][()]))
        res.append(tmp_file.attrs["res"])
        tmp_file.close()
    os.chdir(cwd)
    data = np.squeeze(np.array(data))
    res = np.array(res)

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
    for j in range(data.shape[0]):
        ax[j].imshow(np.abs(data[j][-1]), cmap='gray')
        ax[j].text(
          data[j].shape[-1]-25, data[j].shape[-1]-5, labels[j], color="w")
    plt.savefig("./figures/Heart.png", dpi=300)
    input("Press any Key to exit...")
