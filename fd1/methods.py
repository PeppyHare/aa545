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
