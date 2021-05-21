"""Finite differencing methods."""
import itertools

import numpy as np
import numpy.typing as npt
import numba


@numba.njit(boundscheck=False, nogil=True)
def F(Q: npt.ArrayLike, gamma: float, mu: float):
    """Calculate MHD flux term in the x-direction.

    Given vector Q of conservative ideal MHD variables, returns F such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    F = np.zeros_like(Q)
    # What happens when u[0] == 0? Best not to think about it...
    rvsq = (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / Q[0]  # rho * v^2
    bsqmu = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / mu  # B^2 / mu
    p = (gamma - 1) * (Q[7] - 0.5 * rvsq - 0.5 * bsqmu)  # pressure
    bdotvmu = (Q[1] * Q[4] + Q[2] * Q[5] + Q[3] * Q[6]) / mu  # dot(B, v) / mu
    F[0] += Q[1]
    F[1] += Q[1] ** 2 / Q[0] + p + 0.5 * bsqmu - Q[4] ** 2 / mu
    F[2] += Q[1] * Q[2] / Q[0] - Q[4] * Q[5] / mu
    F[3] += Q[1] * Q[3] / Q[0] - Q[4] * Q[5] / mu
    F[5] += (Q[1] * Q[5] - Q[2] * Q[4]) / Q[0]
    F[6] += (Q[1] * Q[6] - Q[3] * Q[4]) / Q[0]
    F[7] += (Q[7] + p + bsqmu) * Q[1] - Q[4] * bdotvmu

    return F


@numba.njit(boundscheck=False, nogil=True)
def G(Q: npt.ArrayLike, gamma: float, mu: float):
    """Calculate MHD flux term in the y-direction.

    Given vector Q of conservative ideal MHD variables, returns G such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    G = np.zeros_like(Q)
    # What happens when u[0] == 0? Best not to think about it...
    rvsq = (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / Q[0]  # rho * v^2
    bsqmu = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / mu  # B^2 / mu
    p = (gamma - 1) * (Q[7] - 0.5 * rvsq - 0.5 * bsqmu)  # pressure
    bdotvmu = (Q[1] * Q[4] + Q[2] * Q[5] + Q[3] * Q[6]) / mu  # dot(B, v) / mu

    G[0] += Q[2]
    G[1] += Q[1] * Q[2] / Q[0] - Q[4] * Q[5] / mu
    G[2] += Q[2] ** 2 / Q[0] + p + bsqmu / 2 - Q[5] ** 2 / mu
    G[3] += Q[2] * Q[3] / mu - Q[5] * Q[6] / mu
    G[4] += (Q[2] * Q[4] - Q[1] * Q[3]) / mu
    G[6] += (Q[2] * Q[6] - Q[1] * Q[5]) / mu
    G[7] += (Q[7] + p + bsqmu / 2) * Q[2] - bdotvmu * Q[5]

    return G


@numba.njit(boundscheck=False, nogil=True)
def H(Q: npt.ArrayLike, gamma: float, mu: float):
    """Calculate MHD flux term in the z-direction.

    Given vector Q of conservative ideal MHD variables, returns H such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    H = np.zeros_like(Q)
    # What happens when u[0] == 0? Best not to think about it...
    rvsq = (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / Q[0]  # rho * v^2
    bsqmu = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / mu  # B^2 / mu
    p = (gamma - 1) * (Q[7] - 0.5 * rvsq - 0.5 * bsqmu)  # pressure
    bdotvmu = (Q[1] * Q[4] + Q[2] * Q[5] + Q[3] * Q[6]) / mu  # dot(B, v) / mu

    H[0] += Q[3]
    H[1] += Q[1] * Q[3] / Q[0] - Q[4] * Q[6] / mu
    H[2] += Q[2] * Q[3] / Q[0] - Q[5] * Q[6] / mu
    H[3] += Q[3] ** 2 / Q[0] + p + bsqmu / 2 - Q[6] ** 2 / mu
    H[4] += Q[4] * Q[3] - Q[1] * Q[6]
    H[5] += Q[5] * Q[3] - Q[2] * Q[6]
    H[7] += (Q[7] + p + bsqmu / 2) * Q[3] - bdotvmu * Q[6]

    return H


@numba.njit(boundscheck=False, parallel=False, nogil=True)
def maccormack_time_step(
    Q: npt.ArrayLike,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    gamma: float,
    mu: float,
    bcx: int,
    bcy: int,
    bcz: int,
):
    pred = np.copy(Q)
    # Predictor step first
    Mx = Q.shape[1]
    My = Q.shape[2]
    Mz = Q.shape[3]
    for i in numba.prange(Mx):
        for j in numba.prange(My):
            for k in numba.prange(Mz):
                # for i, j, k in itertools.product(
                #     numba.prange(Mx),
                #     numba.prange(My, numba.prange(Mz)),
                # ):
                F_ijk = F(Q[:, i, j, k], gamma, mu)
                G_ijk = G(Q[:, i, j, k], gamma, mu)
                H_ijk = H(Q[:, i, j, k], gamma, mu)
                if i == Mx - 1:
                    if bcx == 0:  # periodic BC
                        Qip1 = Q[:, 0, j, k]
                    elif bcx == 1:  # conducting rigid wall BC
                        Qip1 = Q[:, i, j, k]
                        Qip1[1] = 0
                        Qip1[4] = 0
                else:
                    Qip1 = Q[:, i + 1, j, k]
                if j == My - 1:
                    if bcy == 0:  # periodic BC
                        Qjp1 = Q[:, i, 0, k]
                    elif bcy == 1:  # conducting rigid wall BC
                        Qjp1 = Q[:, i, j, k]
                        Qjp1[2] = 0
                        Qjp1[5] = 0
                else:
                    Qjp1 = Q[:, i, j + 1, k]
                if k == Mz - 1:
                    if bcz == 0:  # periodic BC
                        Qkp1 = Q[:, i, j, 0]
                    elif bcz == 1:  # conducting rigid wall BC
                        Qkp1 = Q[:, i, j, k]
                        Qkp1[3] = 0
                        Qkp1[6] = 0
                else:
                    Qkp1 = Q[:, i, j, k + 1]
                F_ip1jk = F(Qip1, gamma, mu)
                G_ijp1k = G(Qjp1, gamma, mu)
                H_ijkp1 = H(Qkp1, gamma, mu)
                pred[:, i, j, k] -= dt * (
                    (F_ip1jk - F_ijk) / dx
                    + (G_ijp1k - G_ijk) / dy
                    + (H_ijkp1 - H_ijk) / dz
                )
        # else:
        #     raise Exception("Unsupported boundary conditions")
        # for j in numba.prange(Q.shape[2]):
        #     if bcy == 0:  # periodic BC
        #         jp1 = (j + 1) % Q.shape[2]
        #     else:
        #         raise Exception("Unsupported boundary conditions")
        #     for k in numba.prange(Q.shape[3]):
        #         if bcz == 0:  # periodic BC
        #             kp1 = (k + 1) % Q.shape[3]
        #         else:
        #             raise Exception("Unsupported boundary conditions")
    # Corrector step
    corr = np.copy(Q)
    for i in numba.prange(Q.shape[1]):
        for j in numba.prange(Q.shape[2]):
            for k in numba.prange(Q.shape[3]):
                if i == 0:
                    if bcx == 0:  # periodic BC
                        Qim1 = pred[:, Mx - 1, j, k]
                    elif bcx == 1:  # conducting rigid wall BC
                        Qim1 = pred[:, i, j, k]
                        Qim1[1] = 0
                        Qim1[4] = 0
                    else:
                        raise Exception("Unsupported boundary conditions")
                else:
                    Qim1 = Q[:, i - 1, j, k]
                if j == 0:
                    if bcy == 0:  # periodic BC
                        Qjm1 = pred[:, i, My - 1, k]
                    elif bcy == 1:  # conducting rigid wall BC
                        Qjm1 = pred[:, i, j, k]
                        Qjm1[2] = 0
                        Qjm1[5] = 0
                    else:
                        raise Exception("Unsupported boundary conditions")
                else:
                    Qjm1 = pred[:, i, j - 1, k]
                if k == 0:
                    if bcz == 0:  # periodic BC
                        Qkm1 = pred[:, i, j, Mz - 1]
                    elif bcz == 1:  # conducting rigid wall BC
                        Qkm1 = pred[:, i, j, k]
                        Qkm1[3] = 0
                        Qkm1[6] = 0
                    else:
                        raise Exception("Unsupported boundary conditions")
                else:
                    Qkm1 = Q[:, i, j, k - 1]
                F_ijk = F(pred[:, i, j, k], gamma, mu)
                F_im1jk = F(Qim1, gamma, mu)
                G_ijk = G(pred[:, i, j, k], gamma, mu)
                G_ijm1k = G(Qjm1, gamma, mu)
                H_ijk = H(pred[:, i, j, k], gamma, mu)
                H_ijkm1 = H(Qkm1, gamma, mu)
                corr[:, i, j, k] += dt * (
                    (F_ijk - F_im1jk) / dx
                    + (G_ijk - G_ijm1k) / dy
                    + (H_ijk - H_ijkm1) / dz
                )
    Q += (pred + corr) / 2 - Q


@numba.njit(boundscheck=False, nogil=True)
def divB(
    B: npt.ArrayLike,
    dx: float,
    dy: float,
    dz: float,
    bcx: int,
    bcy: int,
    bcz: int,
):
    """Finite difference approximation of the divergence.

    div(B) = dBx/dx + dBy/dy + dBz/dz
    dBx/dx ~ (B[i+1] - B[i-1])/(2*∆x)
    dBy/dy ~ (B[j+1] - B[j-1])/(2*∆y)
    dBz/dy ~ (B[k+1] - B[k-1])/(2*∆z)
    """
    Mx = B.shape[1]
    My = B.shape[2]
    Mz = B.shape[3]
    div = np.zeros_like(B[0])
    for i in numba.prange(1, Mx - 1):
        div[i] += (B[0, i + 1] - B[0, i - 1]) / (2 * dx)
    if bcx == 0:  # periodic BC
        div[0] += (B[0, 1] - B[0, Mx - 1]) / (2 * dx)
        div[Mx - 1] += (B[0, 0] - B[0, Mx - 2]) / (2 * dx)
    # elif bcx == 1:  # conducting rigid wall BC
    # assert np.sum(B[0, 0]) < 10 ** -6
    # assert np.sum(B[0, Mx - 1]) < 10 ** -6
    # else:
    #     raise Exception("Unsupported boundary conditions")

    for j in numba.prange(1, My - 1):
        div[:, j] += (B[1, :, j + 1] - B[1, :, j - 1]) / (2 * dy)
    if bcy == 0:  # periodic BC
        div[:, 0] += (B[1, :, 1] - B[1, :, My - 1]) / (2 * dy)
        div[:, My - 1] += (B[1, :, 0] - B[1, :, My - 2]) / (2 * dy)
    # elif bcy == 1:  # conducting rigid wall BC
    # assert np.sum(B[1, :, 0]) < 10 ** -6
    # assert np.sum(B[1, :, My - 1]) < 10 ** -6
    # else:
    #     raise Exception("Unsupported boundary conditions")

    for k in numba.prange(1, Mz - 1):
        div[:, :, k] += (B[2, :, :, k + 1] - B[2, :, :, k - 1]) / (2 * dz)
    if bcz == 0:  # periodic BC
        div[:, :, 0] += (B[2, :, :, 1] - B[2, :, :, Mz - 1]) / (2 * dz)
        div[:, :, Mz - 1] += (B[2, :, :, 0] - B[2, :, :, Mz - 2]) / (2 * dz)
    # elif bcz == 1:  # conducting rigid wall BC
    # assert np.sum(B[2, :, :, 0]) < 10 ** -6
    # assert np.sum(B[2, :, :, Mz - 1]) < 10 ** -6
    # else:
    #     raise Exception("Unsupported boundary conditions")
    return div
