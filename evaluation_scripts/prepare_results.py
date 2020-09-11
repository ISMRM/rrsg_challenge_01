#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 08:03:39 2020

@author: omaier
"""

import numpy as np
import pickle
import h5py
import os
import scipy.io as sio


def read(path=os.getcwd()):
    data = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.pkl' in file:
                if not str(root.split(os.sep)[-1]) in data.keys():
                    data[str(root.split(os.sep)[-1])] = {}
                data[
                    str(root.split(os.sep)[-1])
                    ][
                        str(file.split('.')[0])
                        ] = read_pkl(
                    root,
                    file
                    )
            if '.h5' in file:
                if "maier" in root:
                    if "TUG_maier" not in data.keys():
                        data["TUG_maier"] = {}
                    key = "TUG_maier"
                    key_dat = str(file.split('.')[0])
                elif "hammernik" in root:
                    if "TUG_hammernik" not in data.keys():
                        data["TUG_hammernik"] = {}
                    key = "TUG_hammernik"
                    key_dat = str(root.split(os.sep)[-1])
                elif "Ref" in root:
                    if "Ref_python" not in data.keys():
                        data["Ref_python"] = {}
                    key = "Ref_python"
                    key_dat = str(file.split('.')[0])
                else:
                    continue
                data[key][
                         key_dat
                        ] = read_h5(
                    root,
                    file
                    )

            if '.mat' in file:
                if not str(root.split(os.sep)[-1]) in data.keys():
                    data[str(root.split(os.sep)[-1])] = {}
                data[
                    str(root.split(os.sep)[-1])
                    ][
                        str(file.split('.')[0])
                        ] = read_mat(
                    root,
                    file
                    )
    return data


def read_pkl(root, fname):
    f = open(root+os.sep+fname, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def read_h5(root, fname):
    f = h5py.File(root+os.sep+fname, 'r')
    data = {}
    for key in f.keys():
        data[key] = f[key][()]
    f.close()
    return data


def read_mat(root, fname):
    data = {}
    sio.loadmat(root+os.sep+fname, data)
    return data


def sort(data):
    sorted_data = {}
    for key, value in data.items():
        if key not in sorted_data.keys():
            sorted_data[key] = {}
        if "NYU" in key:
            sorted_data[key] = prepare_NYU_data(value)
        if "Ludger" in key:
            sorted_data[key] = prepare_BUFF_data(value)
        if "Karolinska" in key:
            sorted_data[key] = prepare_Karolinska_data(value)
        if "Utah" in key:
            sorted_data[key] = prepare_Utah_data(value)
        if "ETH" in key:
            sorted_data[key] = prepare_ETH_data(value)
        if "SCU" in key:
            sorted_data[key] = prepare_USC_data(value)
        if "maier" in key:
            sorted_data[key] = prepare_TUG_maier_data(value)
        if "Eindhoven" in key:
            sorted_data[key] = prepare_Eindhoven_data(value)
        if "Berkeley" in key:
            sorted_data[key] = prepare_Berkeley_data(value)
        if "Stanford" in key:
            sorted_data[key] = prepare_Stanford_data(value)
        if "hammernik" in key:
            sorted_data[key] = prepare_TUG_hammernik_data(value)
        if "Ref_python" in key:
            sorted_data[key] = prepare_Ref_data_python(value)
        if "Ref_matlab" in key:
            sorted_data[key] = prepare_Ref_data_matlab(value)
    return sorted_data


def prepare_NYU_data(data):
    prepdat = {}
    prepdat["brain"] = {}
    prepdat["heart"] = {}
    for key, value in data.items():
        if "heart" in key:
            prepdat["heart"] = np.squeeze(
                np.flip(
                    np.transpose(
                        value["img"],
                        (3, 2, 0, 1)
                    )[:, -1, ...], -2))
        if "head" in key:
            prepdat["brain"] = np.flip(np.flip(
                np.transpose(value["img"], (3, 2, 0, 1)),
                -2), -1)
    return prepdat


def prepare_BUFF_data(data):
    prepdat = {}
    prepdat["brain"] = {}
    prepdat["heart"] = {}
    for key, value in data.items():
        if "Cardio" in key:
            tmp_dat = np.zeros((4,)+value["sumOfSquaresReco"].shape,
                               dtype=np.complex128)
            for j in range(value["recos"].shape[1]):
                tmp_dat[j] = value["recos"][0, j]
            prepdat["heart"] = tmp_dat
        if "Brain" in key:
            tmp_dat = np.zeros((4, 3)+value["sumOfSquaresReco"].shape,
                               dtype=np.complex128)
            for j in range(4):
                tmp_dat[j, 0] = value["singleCoilExamples"][..., j]
                tmp_dat[j, 1] = value["recoOne"][..., j]
                tmp_dat[j, 2] = value["recoTol"][..., j]
            prepdat["brain"] = np.flip(np.flip(tmp_dat, -1), -2)
    prepdat["heart"] = np.flip(prepdat["heart"], -2)
    return prepdat


def prepare_Karolinska_data(data):
    prepdat = {}
    prepdat["brain"] = {}
    prepdat["heart"] = {}
    for key, value in data.items():
        if "heart" in key:
            tmp_dat = np.zeros((4,)+value["cardiac_11"].shape,
                               dtype=np.complex128)
            ind = 3
            for key2, value2 in value.items():
                if "cardiac" in key2:
                    tmp_dat[ind] = value2
                    ind -= 1
            prepdat["heart"] = tmp_dat
        if "head" in key:
            if not np.any(prepdat["brain"]):
                tmp_dat = np.zeros((4, 3, 300, 300),
                                   dtype=np.complex128)
                prepdat["brain"] = tmp_dat
            if "SENSE" in key:
                for j in range(4):
                    prepdat["brain"][j, 1] = value["img_sense_iter"][
                        ..., 0, j]
                    prepdat["brain"][j, 2] = value["img_sense_iter"][
                        ..., -1, j]
            else:
                for j in range(4):
                    prepdat["brain"][j, 0] = value["img_coils"][..., 0, j]
    prepdat["heart"] = np.flip(prepdat["heart"], -2)
    return prepdat

def prepare_Utah_data(data):
    prepdat = {}
    prepdat["brain"] = {}
    prepdat["heart"] = {}
    for key, value in data.items():
        tmp_dat_car = np.zeros((4,)+value["Image_11_rays"].shape,
                               dtype=np.complex128)
        tmp_dat = np.zeros((4, 3)+value["Image_24_rays"].shape[:-1],
                           dtype=np.complex128)

        ind = [55, 33, 22, 11]
        ind2 = (96/np.array([1, 2, 3, 4])).astype(int)
        for j in range(4):
            tmp_dat_car[j] = value["Image_"+str(ind[j])+"_rays"]
            tmp_dat[j, 0] = value["Image_single_coil_"+str(ind2[j])+"_rays"]
            tmp_dat[j, 1] = value["Image_"+str(ind2[j])+"_rays"][..., 0]
            tmp_dat[j, 2] = value["Image_"+str(ind2[j])+"_rays"][..., -1]

        prepdat["heart"] = tmp_dat_car
        prepdat["brain"] = tmp_dat
    prepdat["brain"] = np.flip(prepdat["brain"], -1)
    return prepdat


def prepare_ETH_data(data):
    prepdat = {}
    prepdat["brain"] = {}
    prepdat["heart"] = {}
    for key, value in data.items():
        if "heart" in key:
            tmp_dat = np.zeros((4,)+value["cardioImages"].shape[:-1],
                               dtype=np.complex128)
            ind = 3
            for j in range(4):
                if j == 0:
                    ind2 = 4
                else:
                    ind2 = ind
                tmp_dat[ind] = value["cardioImages"][..., ind2]
                ind -= 1
            prepdat["heart"] = tmp_dat
        if "head" in key:
            if "SENSE" in key:
                if not np.any(prepdat["brain"]):
                    tmp_dat = np.zeros(
                        (4, 3)+value["outSense"][0, 0][1].shape,
                        dtype=np.complex128)
                    prepdat["brain"] = tmp_dat
                for j in range(4):
                    prepdat["brain"][j, 1] = value["outSense"][0, j][-1][0, 0]
                    prepdat["brain"][j, 2] = value["outSense"][0, j][1]
            else:
                if not np.any(prepdat["brain"]):
                    tmp_dat = np.zeros(
                        (4, 3)+value["outSingle"][0, 0][1].shape,
                        dtype=np.complex128)
                    prepdat["brain"] = tmp_dat
                for j in range(4):
                    prepdat["brain"][j, 0] = value["outSingle"][0, j][0][
                        int(tmp_dat.shape[2]/2):-int(tmp_dat.shape[2]/2),
                        int(tmp_dat.shape[2]/2):-int(tmp_dat.shape[2]/2), 0]
    prepdat["heart"] = np.flip(np.flip(prepdat["heart"], -2), 0)
    prepdat["brain"] = np.flip(np.flip(prepdat["brain"], -2), -1)
    return prepdat

def prepare_USC_data(data):
    prepdat = {}
    prepdat["brain"] = {}
    prepdat["heart"] = {}
    for key, value in data.items():
        if "heart" in key:
            tmp_dat = np.zeros((4,)+value["results"][0, 0][-1].shape,
                               dtype=np.complex128)
            for j in range(4):
                tmp_dat[j] = value["results"][j, 0][-1]
            prepdat["heart"] = tmp_dat
        if "head" in key:
            if not np.any(prepdat["brain"]):
                tmp_dat = np.zeros((4, 3)+value["results"][0, 0][-1].shape,
                                   dtype=np.complex128)
                prepdat["brain"] = tmp_dat
            for j in range(4):
                prepdat["brain"][j, 0] = value["results"][j, 0][-3]
                prepdat["brain"][j, 1] = value["results"][j, 0][-2]
                prepdat["brain"][j, 2] = value["results"][j, 0][-1]

    return prepdat


def prepare_TUG_maier_data(data):
    prepdat = {}
    prepdat["heart"] = np.zeros(
        (4, 240, 240),
        dtype=np.complex128)
    prepdat["brain"] = np.zeros(
        (4, 3, 300, 300),
        dtype=np.complex128)
    ind_heart = 0
    ind_brain = 0
    for key, value in sorted(data.items(), reverse=True):
        if "heart" in key:
            prepdat["heart"][ind_heart] = np.squeeze(value["CG_reco"][-1])
            ind_heart += 1
        if "brain" in key:
            prepdat["brain"][3-ind_brain, 0] = np.squeeze(
                value["images_ifft_coils_"][0, 0, 0])
            prepdat["brain"][3-ind_brain, 1] = np.squeeze(value["CG_reco"][1])
            prepdat["brain"][3-ind_brain, 2] = np.squeeze(value["CG_reco"][-1])
            ind_brain += 1
    prepdat["heart"] = np.flip(prepdat["heart"], -2)
    prepdat["brain"] = np.flip(np.flip(prepdat["brain"], -2), -1)
    return prepdat

def prepare_Eindhoven_data(data):
    prepdat = {}
    prepdat["heart"] = np.zeros(
        (4, 320, 320),
        dtype=np.complex128)
    prepdat["brain"] = np.zeros(
        (4, 3, 512, 512),
        dtype=np.complex128)
    for key, value in sorted(data.items(), reverse=True):
        if "heart" in key:
            for j in range(4):
                prepdat["heart"][j] = value[j+1]["recon_img"][0][-1]
        if "brain" in key:
            for j in range(4):
                prepdat["brain"][j, 0] = value[j+1]["recon_img"][0][2]
                prepdat["brain"][j, 1] = value[j+1]["recon_img"][0][0]
                prepdat["brain"][j, 2] = value[j+1]["recon_img"][0][1]
    prepdat["heart"] = np.flip(prepdat["heart"], -2)
    prepdat["brain"] = np.flip(np.flip(prepdat["brain"], -2), -1)
    return prepdat


def prepare_Berkeley_data(data):
    prepdat = {}
    prepdat["heart"] = np.zeros(
        (4, 300, 300),
        dtype=np.complex128)
    prepdat["brain"] = np.zeros(
        (4, 3, 300, 300),
        dtype=np.complex128)
    ind_heart = 0
    ind_brain = 0
    for key, value in sorted(data.items(), reverse=False):
        if "Heart" in key and "Final" in key:
            if "6" not in key:
                prepdat["heart"][ind_heart] = value
                ind_heart += 1
        if "6" not in key:
            if "Brain" in key and "final" in key:
                prepdat["brain"][ind_brain, 2] = value
                ind_brain += 1
            if "Brain" in key and "Singleiter" in key:
                prepdat["brain"][ind_brain, 1] = value
                ind_brain += 1
            if "Brain" in key and "Single_" in key:
                prepdat["brain"][ind_brain, 0] = value
                ind_brain += 1
        else:
            ind_brain = 0
    prepdat["heart"] = np.flip(prepdat["heart"], -2)
    prepdat["brain"] = np.flip(np.flip(prepdat["brain"], -2), -1)
    return prepdat


def prepare_Stanford_data(data):
    prepdat = {}
    prepdat["heart"] = np.zeros((4,)+data["Results_Heart"][0][3].shape,
                                dtype=np.complex128)
    prepdat["brain"] = np.zeros((4,3)+data["Results_Head"][0][3].shape,
                                dtype=np.complex128)
    for key, value in sorted(data.items(), reverse=True):
        if "Heart" in key:
            for j in range(4):
                prepdat["heart"][j] = np.array(value)[j, 3]
        if "Head" in key:
            for j in range(4):
                prepdat["brain"][j, 0] = np.array(value)[j, 1]
                prepdat["brain"][j, 1] = np.array(value)[j, 2]
                prepdat["brain"][j, 2] = np.array(value)[j, 3]
    prepdat["heart"] = np.flip(prepdat["heart"], -2)
    prepdat["brain"] = np.flip(np.flip(prepdat["brain"], -2), -1)
    return prepdat


def prepare_TUG_hammernik_data(data):
    prepdat = {}
    prepdat["heart"] = np.zeros(
        (4, 240, 240),
        dtype=np.complex128)
    prepdat["brain"] = np.zeros(
        (4, 3, 300, 300),
        dtype=np.complex128)
    ind_heart = 0
    ind_brain = 0
    for key, value in sorted(data.items(), reverse=True):
        if "heart" in key:
            prepdat["heart"][ind_heart] = value["cgsense_ic_dc"][-1]
            ind_heart += 1
        if "brain" in key:
            prepdat["brain"][3-ind_brain, 0] = np.flip(
                value["img_regridded"][..., -1], -2)
            prepdat["brain"][3-ind_brain, 1] = value["cgsense_ic_dc"][0]
            prepdat["brain"][3-ind_brain, 2] = value["cgsense_ic_dc"][-1]
            ind_brain += 1
    prepdat["heart"] = np.flip(prepdat["heart"], -1)
    return prepdat


def prepare_Ref_data_python(data):
    prepdat = {}
    prepdat["heart"] = np.zeros(
        (4, 240, 240),
        dtype=np.complex128)
    prepdat["brain"] = np.zeros(
        (4, 3, 300, 300),
        dtype=np.complex128)
    ind_heart = 0
    ind_brain = 0
    for key, value in sorted(data.items(), reverse=True):
        if "heart" in key:
            prepdat["heart"][3-ind_heart] = np.squeeze(value["CG_reco"][-1])
            ind_heart += 1
        if "brain" in key:
            prepdat["brain"][3-ind_brain, 0] = np.squeeze(
                value["Coil_images"][0])
            prepdat["brain"][3-ind_brain, 1] = np.squeeze(value["CG_reco"][0])
            prepdat["brain"][3-ind_brain, 2] = np.squeeze(value["CG_reco"][-1])
            ind_brain += 1
    prepdat["heart"] = np.flip(prepdat["heart"], -2)
    prepdat["brain"] = np.flip(np.flip(prepdat["brain"], -2), -1)
    return prepdat

def prepare_Ref_data_matlab(data):
    prepdat = {}
    prepdat["brain"] = {}
    prepdat["heart"] = {}
    for key, value in data.items():
        if "heart" in key:
            tmp_dat = np.zeros((4,)+value["cardiacImages"].shape[:-1],
                               dtype=np.complex128)
            ind = 3
            for j in range(4):
                if j == 0:
                    ind2 = 4
                else:
                    ind2 = ind
                tmp_dat[ind] = value["cardiacImages"][..., ind2]
                ind -= 1
            prepdat["heart"] = tmp_dat
        if "brain" in key:
            if not np.any(prepdat["brain"]):
                tmp_dat = np.zeros(
                    (4, 3)+value["outSingle"][0, 0][0, 0][0].shape,
                    dtype=np.complex128)
                prepdat["brain"] = tmp_dat
            for j in range(4):
                prepdat["brain"][j, 0] = value["outSingle"][
                    j, 0][0, 0][-3][0, 0]
                prepdat["brain"][j, 2] = value["outSense"][j, 0][0, 0][0]
                prepdat["brain"][j, 1] = value["outSense"][
                    j, 0][0, 0][-3][0, 0]

    prepdat["heart"] = np.flip(np.flip(prepdat["heart"], -2), 0)
    prepdat["brain"] = np.flip(np.flip(prepdat["brain"], -2), -1)
    return prepdat
