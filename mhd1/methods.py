"""Finite differencing methods."""

import numpy as np
import numpy.typing as npt
import numba


@numba.njit
def exact_advection_periodic(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    # Shift all spatial indices forward by c*t/dx
    M = u_j.size
    for j in range(M):
        j_new = int(j - c * n * dt / dx) % M
        u_j[j] += u_hist[j_new, 0] - u_j[j]

    # Store snapshots of u
    if n % subsample_ratio == 0 and u_hist.shape[1] > 1:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j


@numba.njit
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
    for i in numba.prange(Q.shape[1]):
        if bcx == 0:  # periodic BC
            ip1 = (i + 1) % Q.shape[1]
        else:
            raise Exception("Unsupported boundary conditions")
        for j in numba.prange(Q.shape[2]):
            if bcy == 0:  # periodic BC
                jp1 = (j + 1) % Q.shape[2]
            else:
                raise Exception("Unsupported boundary conditions")
            for k in numba.prange(Q.shape[3]):
                if bcz == 0:  # periodic BC
                    kp1 = (k + 1) % Q.shape[3]
                else:
                    raise Exception("Unsupported boundary conditions")
                pred = np.copy(Q[:, i, j, k])
                corr = np.copy(Q[:, i, j, k])
                F_ijk = F(Q[:, i, j, k], gamma, mu)
                F_ip1jk = F(Q[:, ip1, j, k], gamma, mu)
                G_ijk = G(Q[:, i, j, k], gamma, mu)
                G_ijp1k = G(Q[:, i, jp1, k], gamma, mu)
                H_ijk = H(Q[:, i, j, k], gamma, mu)
                H_ijkp1 = H(Q[:, i, j, kp1], gamma, mu)
                pred[:, i, j, k] -= dt * (
                    (F_ip1jk - F_ijk) / dx
                    + (G_ijp1k - G_ijk) / dy
                    + (H_ijkp1 - H_ijk) / dz
                )
    # Corrector step
    for i in numba.prange(Q.shape[1]):
        if bcx == 0:  # periodic BC
            im1 = (i - 1) % Q.shape[1]
        else:
            raise Exception("Unsupported boundary conditions")
        for j in numba.prange(Q.shape[2]):
            if bcy == 0:  # periodic BC
                jm1 = (j - 1) % Q.shape[2]
            else:
                raise Exception("Unsupported boundary conditions")
            for k in numba.prange(Q.shape[3]):
                if bcz == 0:  # periodic BC
                    km1 = (k - 1) % Q.shape[3]
                else:
                    raise Exception("Unsupported boundary conditions")
                F_ijk = F(pred[:, i, j, k], gamma, mu)
                F_im1jk = F(pred[:, im1, j, k], gamma, mu)
                G_ijk = G(pred[:, i, j, k], gamma, mu)
                G_ijm1k = G(pred[:, i, jm1, k], gamma, mu)
                H_ijk = H(pred[:, i, j, k], gamma, mu)
                H_ijkm1 = H(pred[:, i, j, km1], gamma, mu)
                corr = dt * (
                    (F_ijk - F_im1jk) / dx
                    + (G_ijk - G_ijm1k) / dy
                    + (H_ijk - H_ijkm1) / dz
                )
                Q[:, i, j, k] += (pred[:, i, j, k] + corr) / 2 - Q[:, i, j, k]


@numba.njit
def F(Q, gamma, mu):
    """Calculate MHD flux term.

    Given vector Q of conservative ideal MHD variables, returns F such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    F = np.zeros_like(Q)
    # What happens when u[0] == 0???
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


@numba.njit
def G(Q, gamma, mu):
    """Calculate MHD flux term.

    Given vector Q of conservative ideal MHD variables, returns G such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    G = np.zeros_like(Q)
    # What happens when u[0] == 0???
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


@numba.njit
def H(Q, gamma, mu):
    """Calculate MHD flux term.

    Given vector Q of conservative ideal MHD variables, returns H such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    H = np.zeros_like(Q)
    # What happens when u[0] == 0???
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


@numba.njit
def exact_advection_dirichlet(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    # Shift all spatial indices forward by c*t/dx
    M = u_j.size
    for j in range(M):
        j_new = int(j - c * n * dt / dx)
        if j_new <= M and j_new > 0:
            u_j[j] += u_hist[j_new, 0] - u_j[j]
        else:
            u_j[j] -= u_j[j]

    # Store snapshots of u
    if n % subsample_ratio == 0 and u_hist.shape[1] > 1:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j


@numba.njit
def time_step_pure_advection(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    """Exact solution to the advection equation."""
    if c * n * dt % dx < c * dt:
        for j in numba.prange(1, u_j.size - 1):
            u_j[-j] = u_j[-j - 1]

    # Store snapshots of u
    if n % subsample_ratio == 0:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j


@numba.njit
def time_step_ftcs_dirichlet(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    """Forward-time, centered-space.

    Dirichlet boundary conditions:
        u_j[0] = u_j[0, t=0]
        u_j[M] = u_j[M, t=0]
    """
    cfl = c * dt / dx
    j = np.arange(1, u_j.size - 1)
    u_j[j] += 0.5 * cfl * (u_j[j - 1] - u_j[j + 1])

    # Store snapshots of u
    if n % subsample_ratio == 0:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j


@numba.njit
def time_step_ftcs_periodic(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    """Forward-time, centered-space.

    Dirichlet boundary conditions:
        u_j[0] = u_j[0, t=0]
        u_j[M] = u_j[M, t=0]
    """
    cfl = c * dt / dx
    u_j[0] += 0.5 * cfl * (u_j[-1] - u_j[1])
    for j in numba.prange(1, u_j.size - 1):
        u_j[j] += 0.5 * cfl * (u_j[j - 1] - u_j[j + 1])
    u_j[-1] += 0.5 * cfl * (u_j[-2] - u_j[0])

    # Store snapshots of u
    if n % subsample_ratio == 0:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j


@numba.njit
def time_step_simple_upwind_dirichlet(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    """Simple upwind for positive velocity.

    Dirichlet boundary conditions:
        u_j[0] = u_j[0, t=0]
        u_j[M] = u_j[M, t=0]
    """
    cfl = c * dt / dx
    j = np.arange(1, u_j.size - 1)
    if c > 0:
        u_j[j] += cfl * (u_j[j - 1] - u_j[j])
    else:
        u_j[j] += cfl * (u_j[j] - u_j[j + 1])

    # Store snapshots of u
    if n % subsample_ratio == 0:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j


@numba.njit
def time_step_lax_dirichlet(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    """Lax algorithm.

    Dirichlet boundary conditions:
        u_j[0] = u_j[0, t=0]
        u_j[M] = u_j[M, t=0]
    """
    cfl = c * dt / dx
    j = np.arange(1, u_j.size - 1)
    u_j[j] += 0.5 * cfl * (u_j[j - 1] - u_j[j + 1]) + 0.5 * (
        u_j[j + 1] - 2 * u_j[j] + u_j[j - 1]
    )

    # Store snapshots of u
    if n % subsample_ratio == 0:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j


@numba.njit
def time_step_lax_wendroff_dirichlet(
    u_j: npt.ArrayLike,
    u_hist: npt.ArrayLike,
    n: int,
    subsample_ratio: int,
    dx: float,
    dt: float,
    c: float,
):
    """Lax-Wendroff algorithm.

    Dirichlet boundary conditions:
        u_j[0] = u_j[0, t=0]
        u_j[M] = u_j[M, t=0]
    """
    cfl = c * dt / dx
    j = np.arange(1, u_j.size - 1)
    u_j[j] += 0.5 * cfl * (u_j[j - 1] - u_j[j + 1]) + 0.5 * cfl ** 2 * (
        u_j[j + 1] - 2 * u_j[j] + u_j[j - 1]
    )

    # Store snapshots of u
    if n % subsample_ratio == 0:
        sampled_n = int(n / subsample_ratio)
        if sampled_n > 0:
            u_hist[:, sampled_n] += u_j
