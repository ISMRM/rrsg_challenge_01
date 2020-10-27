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


def nlinvns(Y, n, returnProfiles=True, realConstr=False):
    r"""
    Compute coil sensitivity profiles using NLINV.

    Args
    ----
      Y (numpy.array):
        k-space data
      n (int):
        Number of linearization steps
      returnProfiles (bool):
        Return sensitivity profiles
      realConstr (bool):
        Real value constraint on the reconstructed image

    Returns
    -------
      numpy.array
        The estimated indensity corrected image (position 0)\n
        The estimated image without correction (position 1)\n
        The estimated coils if returnProfiles is True starting at
        position 2 to end.
    """
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

    # initialize mask and _weights
    P = np.ones(Y[0, :, :].shape, dtype=np.complex64)
    P[Y[0, :, :] == 0] = 0

    W = _weights(x, y)

    W = np.fft.fftshift(W, axes=(-2, -1))

    # normalize data vector
    y_scale = 100 / np.sqrt(_scal(Y, Y))
    YS = Y * y_scale

    XT = np.zeros([c + 1, y, x], dtype=np.complex64)
    XN = np.copy(X0)

    start = time.perf_counter()
    for i in range(0, n):

        # the application of the _weights matrix to XN
        # is moved out of the operator and the derivative
        XT[0, :, :] = np.copy(XN[0, :, :])
        XT[1:, :, :] = _ap_weightsns(W, np.copy(XN[1:, :, :]))

        RES = (YS - _opns(P, XT))

        print(np.round(np.linalg.norm(RES)))

        # calculate rhs
        r = _derHns(P, W, XT, RES, realConstr)

        r = np.array(r + alpha * (X0 - XN), dtype=np.complex64)

        z = np.zeros_like(r)
        d = np.copy(r)
        dnew = np.linalg.norm(r)**2
        dnot = np.copy(dnew)

        for j in range(0, 500):

            # regularized normal equations
            q = _derHns(P, W, XT, _derns(P, W, XT, d), realConstr) + alpha * d
            np.nan_to_num(q)

            a = dnew / np.real(_scal(d, q))
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

        CR = _ap_weightsns(W, XN[1:, :, :])

        if returnProfiles:
            R[2:, i, :, :] = CR / y_scale

        C = (np.conj(CR) * CR).sum(0)

        R[0, i, :, :] = (XN[0, :, :] * np.sqrt(C) / y_scale)
        R[1, i, :, :] = np.copy(XN[0, :, :])

    R = (R)
    end = time.perf_counter()  # sec.process time
    print('done in', round((end - start)), 's')
    return R


def _scal(a, b):
    v = np.array(np.sum(np.conj(a) * b), dtype=np.complex64)
    return v


def _ap_weightsns(W, CT):
    C = _nsIfft(W * CT)
    return C


def _ap_weightsnsH(W, CT):
    C = np.conj(W) * _nsFft(CT)
    return C


def _opns(P, X):
    K = np.array(X[0, :, :] * X[1:, :, :], dtype=np.complex64)
    K = np.array(P * _nsFft(K), dtype=np.complex64)
    return K


def _derns(P, W, X0, DX):
    K = X0[0, :, :] * _ap_weightsns(W, DX[1:, :, :])
    K = K + (DX[0, :, :] * X0[1:, :, :])
    K = P * _nsFft(K)
    return K


def _derHns(P, W, X0, DK, realConstr):
    K = _nsIfft(P * DK)

    if realConstr:
        DXrho = np.sum(np.real(K * np.conj(X0[1:, :, :])), 0)
    else:
        DXrho = np.sum(K * np.conj(X0[1:, :, :]), 0)

    DXc = _ap_weightsnsH(W, (K * np.conj(X0[0, :, :])))
    DX = np.array(np.concatenate(
        (DXrho[None, ...], DXc), axis=0), dtype=np.complex64)
    return DX


def _nsFft(M):
    si = M.shape
    a = 1 / (np.sqrt((si[M.ndim - 1])) * np.sqrt((si[M.ndim - 2])))
    K = np.array((pyfftw.interfaces.numpy_fft.fft2(
        M, norm=None)).dot(a), dtype=np.complex64)
    return K


def _nsIfft(M):
    si = M.shape
    a = np.sqrt(si[M.ndim - 1]) * np.sqrt(si[M.ndim - 2])
    K = np.array(pyfftw.interfaces.numpy_fft.ifft2(M, norm=None).dot(a))
    return K  # .T


def _weights(x, y):
    W = np.zeros([x, y])
    for i in range(0, x):
        for j in range(0, y):
            d = ((i) / x - 0.5)**2 + ((j) / y - 0.5)**2
            W[j, i] = 1 / (1 + 220 * d)**16
    return W
