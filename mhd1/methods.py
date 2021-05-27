"""Finite differencing methods."""

import numpy as np
import numpy.typing as npt
import numba


@numba.njit(boundscheck=True, nogil=True, parallel=False, fastmath=True)
def F(Q: npt.ArrayLike):
    """Calculate MHD flux term in the x-direction.

    Given vector Q of conservative ideal MHD variables, returns F such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    F = np.zeros_like(Q)
    # gamma = 5 / 3  # For 3D problems
    gamma = 2  # For 1D problems
    mu = 1.0
    # What happens when Q[0] == 0? Best not to think about it...
    rvsq = (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / Q[0]  # rho * v^2
    bsqmu = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / mu  # B^2 / mu
    p = (gamma - 1) * (Q[7] - 0.5 * rvsq - 0.5 * bsqmu)  # pressure
    bdotvmu = (Q[1] * Q[4] + Q[2] * Q[5] + Q[3] * Q[6]) / (
        Q[0] * mu
    )  # dot(B, v) / mu
    F[0] += Q[1]
    F[1] += Q[1] ** 2 / Q[0] - Q[4] ** 2 / mu + p + 0.5 * bsqmu
    F[2] += Q[1] * Q[2] / Q[0] - Q[4] * Q[5] / mu
    F[3] += Q[1] * Q[3] / Q[0] - Q[4] * Q[5] / mu
    F[5] += (Q[1] * Q[5] - Q[2] * Q[4]) / Q[0]
    F[6] += (Q[1] * Q[6] - Q[3] * Q[4]) / Q[0]
    F[7] += (Q[7] + p + bsqmu) * Q[1] / Q[0] - bdotvmu * Q[4]

    return F


@numba.njit(
    boundscheck=True, nogil=True, cache=True, parallel=False, fastmath=True
)
def G(Q: npt.ArrayLike):
    """Calculate MHD flux term in the y-direction.

    Given vector Q of conservative ideal MHD variables, returns G such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    G = np.zeros_like(Q)
    # gamma = 5 / 3  # For 3D problems
    gamma = 2  # For 1D problems
    # return G  # Useful short-circuit for decreasing dimensionality
    mu = 1.0
    # What happens when Q[0] == 0? Best not to think about it...
    rvsq = (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / Q[0]  # rho * v^2
    bsqmu = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / mu  # B^2 / mu
    p = (gamma - 1) * (Q[7] - 0.5 * rvsq - 0.5 * bsqmu)  # pressure
    bdotvmu = (Q[1] * Q[4] + Q[2] * Q[5] + Q[3] * Q[6]) / (
        Q[0] * mu
    )  # dot(B, v) / mu

    G[0] += Q[2]
    G[1] += Q[1] * Q[2] / Q[0] - Q[4] * Q[5] / mu
    G[2] += Q[2] ** 2 / Q[0] - Q[5] ** 2 / mu + p + bsqmu / 2
    G[3] += Q[2] * Q[3] / Q[0] - Q[5] * Q[6] / mu
    G[4] += (Q[2] * Q[4] - Q[1] * Q[5]) / Q[0]
    G[6] += (Q[2] * Q[6] - Q[3] * Q[5]) / Q[0]
    G[7] += (Q[7] + p + bsqmu / 2) * Q[2] / Q[0] - bdotvmu * Q[5]

    return G


@numba.njit(
    boundscheck=True, nogil=True, cache=True, parallel=False, fastmath=True
)
def H(Q: npt.ArrayLike):
    """Calculate MHD flux term in the z-direction.

    Given vector Q of conservative ideal MHD variables, returns H such that:

    dQ/dt + dF/dx + dG/dy + dH/dz = 0
    """
    H = np.zeros_like(Q)
    # gamma = 5 / 3  # For 3D problems
    gamma = 2  # For 1D problems
    # return H  # Useful short-circuit for decreasing dimensionality
    mu = 1.0
    # What happens when Q[0] == 0? Best not to think about it...
    rvsq = (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / Q[0]  # rho * v^2
    bsqmu = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / mu  # B^2 / mu
    p = (gamma - 1) * (Q[7] - 0.5 * rvsq - 0.5 * bsqmu)  # pressure
    bdotvmu = (Q[1] * Q[4] + Q[2] * Q[5] + Q[3] * Q[6]) / (
        Q[0] * mu
    )  # dot(B, v) / mu

    H[0] += Q[3]
    H[1] += Q[1] * Q[3] / Q[0] - Q[4] * Q[6] / mu
    H[2] += Q[2] * Q[3] / Q[0] - Q[5] * Q[6] / mu
    H[3] += Q[3] ** 2 / Q[0] - Q[6] ** 2 / mu + p + bsqmu / 2
    H[4] += (Q[4] * Q[3] - Q[1] * Q[6]) / Q[0]
    H[5] += (Q[5] * Q[3] - Q[2] * Q[6]) / Q[0]
    H[7] += (Q[7] + p + bsqmu / 2) * Q[3] / Q[0] - bdotvmu * Q[6]

    return H


@numba.njit(
    boundscheck=True, nogil=True, cache=True, parallel=False, fastmath=True
)
def maccormack_time_step(
    Q: npt.ArrayLike, dx: float, dy: float, dz: float, dt: float
):
    """MacCormack Lax-Wendroff predictor-corrector step.

    Assumes conducting wall boundary conditions for x and y, and periodic
    boundary conditions for z.
    """
    Mx = Q.shape[1]
    My = Q.shape[2]
    Mz = Q.shape[3]
    # Predictor step first
    pred = np.copy(Q)
    # for i in numba.prange(0, Mx - 1):
    for i in range(0, Mx - 1):
        pred[:, i, :, :] -= dt / dx * (F(Q[:, i + 1, :, :]) - F(Q[:, i, :, :]))
    # Conducting walls in x
    pred[1, 0, :, :] = 0
    pred[1, Mx - 1, :, :] = 0
    pred[4, 0, :, :] = Q[4, 0, :, :]
    pred[4, Mx - 1, :, :] = Q[4, Mx - 1, :, :]
    # for j in numba.prange(0, My - 1):
    for j in range(0, My - 1):
        pred[:, :, j, :] -= dt / dy * (F(Q[:, :, j + 1, :]) - F(Q[:, :, j, :]))
    # Conducting walls in y
    pred[2, :, 0, :] = 0
    pred[2, :, My - 1, :] = 0
    pred[5, :, 0, :] = Q[5, :, 0, :]
    pred[5, :, My - 1, :] = Q[5, :, My - 1, :]
    # for k in numba.prange(0, Mz - 1):
    for k in range(0, Mz - 1):
        pred[:, :, :, k] -= dt / dz * (F(Q[:, :, :, k + 1]) - F(Q[:, :, :, k]))
    # Periodic boundary in z
    pred[:, :, :, Mz - 1] -= dt / dz * (F(Q[:, :, :, 0]) - F(Q[:, :, :, k]))

    # Corrector step
    corr = np.copy(Q)
    # for i2 in numba.prange(1, Mx):
    for i2 in range(1, Mx):
        corr[:, i2, :, :] -= (
            dt / dx * (F(pred[:, i2, :, :]) - F(pred[:, i2 - 1, :, :]))
        )
    # Conducting walls in x
    corr[1, 0, :, :] = 0
    corr[1, Mx - 1, :, :] = 0
    corr[4, 0, :, :] = Q[4, 0, :, :]
    corr[4, Mx - 1, :, :] = Q[4, Mx - 1, :, :]
    # for j2 in numba.prange(1, My):
    for j2 in range(1, My):
        corr[:, :, j2, :] -= (
            dt / dy * (F(pred[:, :, j2, :]) - F(pred[:, :, j2 - 1, :]))
        )
    # Conducting walls in y
    corr[2, :, 0, :] = 0
    corr[2, :, My - 1, :] = 0
    corr[5, :, 0, :] = Q[5, :, 0, :]
    corr[5, :, My - 1, :] = Q[5, :, My - 1, :]
    # for k2 in numba.prange(1, Mz):
    for k2 in range(1, Mz):
        corr[:, :, :, k2] -= (
            dt / dz * (F(pred[:, :, :, k2]) - F(pred[:, :, :, k2 - 1]))
        )
    # Periodic boundary in z
    corr[:, :, :, 0] -= (
        dt / dz * (F(pred[:, :, :, 0]) - F(pred[:, :, :, Mz - 1]))
    )
    diffusion_method = "post_avg"
    visc = 0.1 * dx ** 2 / dt
    # Add artificial diffusion
    if diffusion_method == "pre_avg":
        # for i in numba.prange(1, Mx - 1):
        for i in range(1, Mx - 1):
            pred[:, i, :, :] += (
                visc
                * dt
                / dx ** 2
                * (
                    pred[:, i + 1, :, :]
                    - 2 * pred[:, i, :, :]
                    + pred[:, i - 1, :, :]
                )
            )
            corr[:, i, :, :] += (
                visc
                * dt
                / dx ** 2
                * (
                    corr[:, i + 1, :, :]
                    - 2 * corr[:, i, :, :]
                    + corr[:, i - 1, :, :]
                )
            )
        # for j in numba.prange(0, My - 1):
        for j in range(0, My - 1):
            pred[:, :, j, :] += (
                visc
                * dt
                / dy ** 2
                * (
                    pred[:, :, j + 1, :]
                    - 2 * pred[:, :, j, :]
                    + pred[:, :, j - 1, :]
                )
            )
            corr[:, :, j, :] += (
                visc
                * dt
                / dy ** 2
                * (
                    corr[:, :, j + 1, :]
                    - 2 * corr[:, :, j, :]
                    + corr[:, :, j - 1, :]
                )
            )
        # for k in numba.prange(0, Mz):
        for k in range(0, Mz):
            pred[:, :, k, :] += (
                visc
                * dt
                / dz ** 2
                * (
                    pred[:, :, :, ((k + 1) % Mz)]
                    - 2 * pred[:, :, :, k]
                    + pred[:, :, :, ((k - 1) % Mz)]
                )
            )
            corr[:, :, k, :] += (
                visc
                * dt
                / dz ** 2
                * (
                    corr[:, :, :, ((k + 1) % Mz)]
                    - 2 * corr[:, :, :, k]
                    + corr[:, :, :, ((k - 1) % Mz)]
                )
            )

    Q += (pred + corr) / 2 - Q
    if diffusion_method == "post_avg":
        # for i in numba.prange(1, Mx - 1):
        for i in range(1, Mx - 1):
            Q[:, i, :, :] += (
                visc
                * dt
                / dx ** 2
                * (Q[:, i + 1, :, :] - 2 * Q[:, i, :, :] + Q[:, i - 1, :, :])
            )
        # for j in numba.prange(0, My - 1):
        for j in range(0, My - 1):
            Q[:, :, j, :] += (
                visc
                * dt
                / dy ** 2
                * (Q[:, :, j + 1, :] - 2 * Q[:, :, j, :] + Q[:, :, j - 1, :])
            )
        # for k in numba.prange(0, Mz):
        for k in range(0, Mz):
            Q[:, :, k, :] += (
                visc
                * dt
                / dz ** 2
                * (
                    Q[:, :, :, ((k + 1) % Mz)]
                    - 2 * Q[:, :, :, k]
                    + Q[:, :, :, ((k - 1) % Mz)]
                )
            )


@numba.njit(boundscheck=True, nogil=True, parallel=True, fastmath=True)
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
    # for i in numba.prange(1, Mx - 1):
    for i in range(1, Mx - 1):
        div[i] += (B[0, i + 1] - B[0, i - 1]) / (2 * dx)
    if bcx == 0:  # periodic BC
        div[0] += (B[0, 1] - B[0, Mx - 1]) / (2 * dx)
        div[Mx - 1] += (B[0, 0] - B[0, Mx - 2]) / (2 * dx)

    # for j in numba.prange(1, My - 1):
    for j in range(1, My - 1):
        div[:, j] += (B[1, :, j + 1] - B[1, :, j - 1]) / (2 * dy)
    if bcy == 0:  # periodic BC
        div[:, 0] += (B[1, :, 1] - B[1, :, My - 1]) / (2 * dy)
        div[:, My - 1] += (B[1, :, 0] - B[1, :, My - 2]) / (2 * dy)

    # for k in numba.prange(1, Mz - 1):
    for k in range(1, Mz - 1):
        div[:, :, k] += (B[2, :, :, k + 1] - B[2, :, :, k - 1]) / (2 * dz)
    if bcz == 0:  # periodic BC
        div[:, :, 0] += (B[2, :, :, 1] - B[2, :, :, Mz - 1]) / (2 * dz)
        div[:, :, Mz - 1] += (B[2, :, :, 0] - B[2, :, :, Mz - 2]) / (2 * dz)
    return div


@numba.njit(boundscheck=True, nogil=True, parallel=False, fastmath=True)
def calc_cfl(Q: npt.ArrayLike, dx: float, dy: float, dz: float, dt: float):
    """Calculate the maximium value of the CFL constant in each direction."""

    # The largest wave speed is the fluid velocity plus the fast alfven
    # speed
    # gamma = 5 / 3  # For 3D problems
    gamma = 2  # For 1D problems
    mu = 1.0

    # Square of the Alfven speed
    v_a_sq = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / (Q[0] * mu)
    # Square of the Alfven speed component each x-direction
    v_ax_sq = Q[4] ** 2 / (Q[0] * mu)
    v_ay_sq = Q[5] ** 2 / (Q[0] * mu)
    v_az_sq = Q[6] ** 2 / (Q[0] * mu)

    # Square of the sound speed
    rvsq = (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / Q[0]  # rho * v^2
    bsqmu = (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / mu  # B^2 / mu
    p = (gamma - 1) * (Q[7] - 0.5 * rvsq - 0.5 * bsqmu)  # pressure
    v_s_sq = gamma * p / Q[0]

    # Fast Alfven speed in each direction
    vf_x = np.sqrt(
        0.5
        * (
            v_a_sq
            + v_s_sq
            + np.sqrt((v_a_sq + v_s_sq) ** 2 - 4 * (v_ax_sq * v_s_sq))
        )
    )
    vf_y = np.sqrt(
        0.5
        * (
            v_a_sq
            + v_s_sq
            + np.sqrt((v_a_sq + v_s_sq) ** 2 - 4 * (v_ay_sq * v_s_sq))
        )
    )
    vf_z = np.sqrt(
        0.5
        * (
            v_a_sq
            + v_s_sq
            + np.sqrt((v_a_sq + v_s_sq) ** 2 - 4 * (v_az_sq * v_s_sq))
        )
    )

    cfl_x = np.max(np.abs(Q[1] / Q[0]) + vf_x) * dt / dx
    cfl_y = np.max(np.abs(Q[2] / Q[0]) + vf_y) * dt / dy
    cfl_z = np.max(np.abs(Q[3] / Q[0]) + vf_z) * dt / dz
    return np.array([cfl_x, cfl_y, cfl_z])


@numba.njit(fastmath=True, cache=True, parallel=False)
def linear_mhd_time_step(
    v: npt.ArrayLike,
    b: npt.ArrayLike,
    p: npt.ArrayLike,
    dr: float,
    dz: float,
    dt: float,
    p0: npt.ArrayLike,
    rho0: npt.ArrayLike,
    b0r: npt.ArrayLike,
    b0t: npt.ArrayLike,
    b0z: npt.ArrayLike,
    db0rdr: npt.ArrayLike,
    db0rdz: npt.ArrayLike,
    db0tdr: npt.ArrayLike,
    db0tdz: npt.ArrayLike,
    db0zdr: npt.ArrayLike,
    db0zdz: npt.ArrayLike,
):
    """Evolve the time-dependent linear MHD equations forward in time.

    [b0r, b0t, b0z]: Equilibrium magnetic field
    [db0rdr, db0tdr, db0zdr]: Gradient of b0 in the radial direction
    [db0zdr, db0zdt, db0zdz]: Gradient of b0 in the z direction

    This time step is implemented for a pressure-less, cylindrical geometry. The
    equilibrium pressure is zero, and it is assumed that b0 is axially
    symmetric, that is, the gradient in the theta direction is zero.
    """

    Mr = v.shape[1]
    Mz = v.shape[2]

    # TODO: parameterize azimuthal mode
    m = 1

    # TODO: parameterize mu0
    mu = 1.0

    # TODO: parameterize gamma
    gamma = 5 / 3

    dv = np.zeros_like(v)
    db = np.zeros_like(b)
    dp = np.zeros_like(p)
    for j in range(0, Mr):
        r = j * dr
        for k in range(0, Mz):
            # Boundary!
            if j == 0:
                pass
            elif j == Mr-1:
                pass
            else:
                dv[0, j, k] += (
                    dt
                    / rho0[j, k]
                    * (
                        (p[j + 1, k] - p[j - 1, k]) / (2 * dr)
                        + 1
                        / (mu * r)
                        * (
                            (r * b[2, j, k]) * (db0rdz[j, k] - db0zdr[j, k])
                            - (r * b[2, j, k] * db0tdr[j, k])
                            + (r * b0z[j, k])
                            * (
                                (b[0, j, k + 1] - b[0, j, k - 1]) / (2 * dz)
                                - (b[2, j + 1, k] - b[2, j - 1, k]) / (2 * dr)
                            )
                            + b0t[j, k]
                            * (
                                -2 * b[1, j, k]
                                + 1j * m * b[0, j, k]
                                - r * (b[1, j + 1, k] - b[1, j - 1, k]) / (2 * dr)
                            )
                        )
                    )
                )
                dv[1, j, k] += (dt / r) * (
                    1j * m * p[j, k]
                    + 1
                    / mu
                    * (
                        b[2, j, k] * r * db0tdz[j, k]
                        + b0r[j, k] * (b[2, j, k] - 1j * m * b[0, j, k])
                        + b0z[j, k]
                        * (
                            r * (b[1, j, k + 1] - b[1, j, k - 1]) / (2 * dz)
                            - 1j * m * b[2, j, k]
                        )
                        + r * b[0, j, k] * db0tdr[j, k]
                    )
                )
                dv[2, j, k] += dt * (
                    (p[j, k + 1] - p[j, k - 1]) / (2 * dz)
                    + 1
                    / mu
                    * (
                        b[0, j, k] * (db0zdr[j, k] - db0rdz[j, k])
                        + 1
                        / r
                        * (
                            -b[1, j, k] * r * db0tdz[j, k]
                            + b0t[j, k]
                            * (
                                -r * (b[1, j, k + 1] - b[1, j, k - 1]) / (2 * dz)
                                + 1j * m * b[2, j, k]
                            )
                            + r
                            * b0r[j, k]
                            * (
                                (b[2, j + 1, k] - b[2, j - 1, k]) / (2 * dr)
                                - (b[0, j, k + 1] - b[0, j, k - 1]) / (2 * dz)
                            )
                        )
                    )
                )
                db[0, j, k] += dt * (
                    -v[2, j, k] * db0rdz[j, k]
                    + 1
                    / r
                    * (
                        r * b0z[j, k] * (v[0, j, k + 1] - v[0, j, k - 1]) / (2 * dz)
                        + r * v[0, j, k] * db0zdz[j, k]
                        + b0t[j, k] * 1j * m * v[0, j, k]
                        - b0r[j, k]
                        * (
                            r * (v[2, j, k + 1] - v[2, j, k - 1]) / (2 * dz)
                            + 1j * m * v[1, j, k]
                        )
                    )
                )
                db[1, j, k] += dt * (
                    -(v[2, j, k] * db0tdz[j, k])
                    + 1j * m * b0z[j, k] * v[1, j, k]
                    + v[1, j, k] * (db0zdz[j, k] - db0rdr[j, k])
                    - v[0, j, k] * db0tdr[j, k]
                    - b0t[j, k]
                    * (
                        (v[2, j, k + 1] - v[2, j, k - 1]) / (2 * dz)
                        + (v[0, j + 1, k] - v[0, j - 1, k]) / (2 * dr)
                    )
                    + b0r[j, k] * (v[0, j + 1, k] - v[0, j - 1, k]) / (2 * dr)
                )
                db[2, j, k] += (
                    dt
                    / r
                    * (
                        b0t[j, k] * 1j * m * v[0, j, k]
                        + v[0, j, k] * db0rdr[j, k]
                        - r * v[0, j, k] * db0zdr[j, k]
                        + r
                        * b0t[j, k]
                        * (v[2, j + 1, k] - v[2, j - 1, k])
                        / (2 * dr)
                        - b0z[j, k]
                        * (
                            v[0, j, k]
                            + 1j * m * v[1, j, k]
                            + r * (v[0, j + 1, k] - v[0, j - 1, k]) / (2 * dr)
                        )
                    )
                )
                dp[j, k] += (
                    -dt
                    * p0[j, k]
                    * gamma
                    * (
                        ((r + dr) * v[0, j + 1, k] - (r - dr) * v[0, j - 1, k])
                        / (2 * dr * r)
                        + 1j * m * v[1, j, k] / r
                        + (v[2, j, k + 1] - v[2, j, k - 1]) / (2 * dz)
                    )
                )  # Terms proportional to p0 missing here for pressure-free eq.
            v += dv
            b += db
            p += dp
