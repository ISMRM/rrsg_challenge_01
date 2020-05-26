# nlinvns
# % Written and invented
# % by Martin Uecker <muecker@gwdg.de> in 2008-09-22
# %
# % Modifications by Tilman Sumpf 2012 <tsumpf@gwdg.de>:
# %	- removed fftshift during reconstruction (ns = "no shift")
# %	- added switch to return coil profiles
# %	- added switch to force the image estimate to be real
# %	- use of vectorized operation rather than "for" loops
# %
# % Version 0.1
# %
# % Biomedizinische NMR Forschungs GmbH am
# % Max-Planck-Institut fuer biophysikalische Chemie
# Adapted for Python by O. Maier

import numpy as np
import time
import pyfftw


def nlinvns(Y, n, *arg):  # *returnProfiles,**realConstr):
    nrarg = len(arg)
    if nrarg == 2:
        returnProfiles = arg[0]
        realConstr = arg[1]
    elif nrarg < 2:
        realConstr = False
        if nrarg < 1:
            returnProfiles = 0

    print('Start...')

    alpha = 1

    [c, y, x] = Y.shape

    if returnProfiles:
        R = np.zeros([c + 2, n, y, x], complex)

    else:
        R = np.zeros([2, n, y, x], complex)

    # initialization x-vector
    X0 = np.array(np.zeros([c + 1, y, x]), np.complex64)
    X0[0, :, :] = 1

    # initialize mask and weights
    P = np.ones(Y[0, :, :].shape, dtype=np.complex64)
    P[Y[0, :, :] == 0] = 0

    W = weights(x, y)

    W = np.fft.fftshift(W, axes=(-2, -1))

    # normalize data vector
    yscale = 100 / np.sqrt(scal(Y, Y))
    YS = Y * yscale 

    XT = np.zeros([c + 1, y, x], dtype=np.complex64)
    XN = np.copy(X0)

    start = time.perf_counter()
    for i in range(0, n):

        # the application of the weights matrix to XN
        # is moved out of the operator and the derivative
        XT[0, :, :] = np.copy(XN[0, :, :])
        XT[1:, :, :] = apweightsns(W, np.copy(XN[1:, :, :]))

        RES = (YS - opns(P, XT))

        print(np.round(np.linalg.norm(RES))) 

        # calculate rhs
        r = derHns(P, W, XT, RES, realConstr)

        r = np.array(r + alpha * (X0 - XN), dtype=np.complex64)

        z = np.zeros_like(r)
        d = np.copy(r)
        dnew = np.linalg.norm(r)**2
        dnot = np.copy(dnew)

        for j in range(0, 500):

            # regularized normal equations
            q = derHns(P, W, XT, derns(P, W, XT, d), realConstr) + alpha * d
            np.nan_to_num(q)

            a = dnew / np.real(scal(d, q))
            z = z + a * (d)
            r = r - a * q
            np.nan_to_num(r)
            dold = np.copy(dnew)
            dnew = np.linalg.norm(r)**2

            d = d * ((dnew / dold)) + r
            np.nan_to_num(d)
            if (np.sqrt(dnew) < (1e-2 * dnot)):
                break

        print('(', j, ')')

        XN = XN + z

        alpha = alpha / 3

        # postprocessing

        CR = apweightsns(W, XN[1:, :, :])

        if returnProfiles:
            R[2:, i, :, :] = CR / yscale

        C = (np.conj(CR) * CR).sum(0)

        R[0, i, :, :] = (XN[0, :, :] * np.sqrt(C) / yscale)
        R[1, i, :, :] = np.copy(XN[0, :, :])

    R = (R)
    end = time.perf_counter()  # sec.process time
    print('done in', round((end - start)), 's')
    return R


def scal(a, b):
    v = np.array(np.sum(np.conj(a) * b), dtype=np.complex64)
    return v


def apweightsns(W, CT):
    C = nsIfft(W * CT)
    return C


def apweightsnsH(W, CT):
    C = np.conj(W) * nsFft(CT)
    return C


def opns(P, X):
    K = np.array(X[0, :, :] * X[1:, :, :], dtype=np.complex64)
    K = np.array(P * nsFft(K), dtype=np.complex64)
    return K


def derns(P, W, X0, DX):
    K = X0[0, :, :] * apweightsns(W, DX[1:, :, :])
    K = K + (DX[0, :, :] * X0[1:, :, :])
    K = P * nsFft(K)
    return K


def derHns(P, W, X0, DK, realConstr):
    K = nsIfft(P * DK)

    if realConstr:
        DXrho = np.sum(np.real(K * np.conj(X0[1:, :, :])), 0)
    else:
        DXrho = np.sum(K * np.conj(X0[1:, :, :]), 0)

    DXc = apweightsnsH(W, (K * np.conj(X0[0, :, :])))
    DX = np.array(np.concatenate(
        (DXrho[None, ...], DXc), axis=0), dtype=np.complex64)
    return DX


def nsFft(M):
    si = M.shape
    a = 1 / (np.sqrt((si[M.ndim - 1])) * np.sqrt((si[M.ndim - 2])))
    K = np.array((pyfftw.interfaces.numpy_fft.fft2(
        M, norm=None)).dot(a), dtype=np.complex64)
    return K


def nsIfft(M):
    si = M.shape
    a = np.sqrt(si[M.ndim - 1]) * np.sqrt(si[M.ndim - 2])
    K = np.array(pyfftw.interfaces.numpy_fft.ifft2(M, norm=None).dot(a))
    return K  # .T


def weights(x, y):
    W = np.zeros([x, y])
    for i in range(0, x):
        for j in range(0, y):
            d = ((i) / x - 0.5)**2 + ((j) / y - 0.5)**2
            W[j, i] = 1 / (1 + 220 * d)**16
    return W
