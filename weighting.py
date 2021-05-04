"""Weighting functions."""
import math
import time

import numpy as np
import numba
import progressbar


@numba.njit(nogil=True)
def weight_particles(x, gp, dx, M, q=1, order=0):
    """Weight particles to grid. Assume x >= 0.

    Weighting function order determined by value of :order:. Boundary is assumed
    to be periodic such that j = mod(j, m).

    :order: = 0: Nearest-grid-point weighting.

    :order: = 1: Linear weighting. Particles are assigned to the two nearest
    grid points via linear interpolation.
    """
    rho = np.zeros_like(gp)
    eps = 10 ** -8

    # Nearest grid point
    if order == 0:
        for i in numba.prange(x.shape[0]):
            j = round(x[i] / dx)
            rho[j % M] += q / dx

    # Linear weighting
    elif order == 1:
        for i in numba.prange(x.shape[0]):
            xi = x[i] % 1.0
            assert xi >= 0 and xi <= 1
            j = round(xi / dx)  # 0 <= j <= m
            if (j > 0) and (j > M * xi):
                j = j - 1
            assert j <= xi / dx
            # Apply periodic boundary conditions
            if j == M:
                rho[0] += q / dx
            elif j == M - 1:
                rho[j] += q * (gp[j] + dx - xi) / dx ** 2
                rho[0] += q * (xi - gp[j]) / dx ** 2
            else:
                rho[j] += q * (gp[j + 1] - xi) / dx ** 2
                rho[j + 1] += q * (xi - gp[j]) / dx ** 2
            assert (
                abs(np.sum(rho * dx) - q * (i + 1)) < eps
            )  # charge conservation
    else:
        raise ValueError("Incorrect value 'order', must be 0 or 1.")
    assert abs(np.sum(rho * dx) - q * x.size) < eps  # charge conservation
    return rho


@numba.njit(nogil=True)
def weight_particles_new(x, gp, dx, M, q=1, order=0):
    """Weight particles to grid. Assume x >= 0.

    Weighting function order determined by value of :order:. Boundary is assumed
    to be periodic such that j = mod(j, m).

    :order: = 0: Nearest-grid-point weighting.

    :order: = 1: Linear weighting. Particles are assigned to the two nearest
    grid points via linear interpolation.
    """
    rho = np.zeros_like(gp)

    # Nearest grid point
    if order == 0:
        for i in numba.prange(x.shape[0]):
            j = round(x[i] / dx)
            rho[j % M] += q / dx

    # Linear weighting
    elif order == 1:
        for i in numba.prange(x.shape[0]):
            j = math.floor(x[i] * M + 10 ** -8)  # 0 <= j <= m
            # Apply periodic boundary conditions
            if j == M:
                rho[0] += q / dx
            elif j == M - 1:
                rho[j] += q * (gp[j] + dx - x[i]) / dx ** 2
                rho[0] += q * (x[i] - gp[j]) / dx ** 2
            else:
                rho[j] += q * (gp[j + 1] - x[i]) / dx ** 2
                rho[j + 1] += q * (x[i] - gp[j]) / dx ** 2
            assert (
                abs(np.sum(rho * dx) - q * (i + 1)) < 10 ** -6
            )  # charge conservation
    else:
        raise ValueError("Incorrect value 'order', must be 0 or 1.")
    assert abs(np.sum(rho * dx) - q * x.size) < 10 ** -6  # charge conservation
    return rho


@numba.njit(nogil=True)
def weight_field(x, gp, e_j, dx, order=0):
    """Obtain weighted field on particle from the grid.

    Weighting function order determined by value of :order:. Boundary is assumed
    to be periodic such that j = mod(j, m).

    :order: = 0: Nearest-grid-point weighting. Force on particle is given by the
    value of the field at the nearest grid point.

    :order: = 1: Linear weighting. Force on a particle is given by linear
    interpolation between the two nearest grid points.
    """
    e_i = np.zeros_like(x)
    m = gp.shape[0]

    for i, x_i in enumerate(x):
        # Nearest grid point
        if order == 0:
            j = round(x_i / dx)
            e_i[i] = e_j[j % m]

        # Linear
        elif order == 1:
            j = round(x[i] * m)  # 0 <= j <= m
            if j > 0 and j / m > x[i]:
                j = j - 1
            # Apply periodic boundary conditions
            if j == m:
                e_i[i] = e_j[0]
            elif j == m - 1:
                e_i[i] = (
                    (gp[j] + dx - x_i) * e_j[j] + (x_i - gp[j]) * e_j[0]
                ) / dx
            else:
                e_i[i] = (
                    (gp[j + 1] - x_i) * e_j[j] + (x_i - gp[j]) * e_j[j + 1]
                ) / dx

        else:
            raise ValueError("Incorrect value 'order', must be 0 or 1.")

    return e_i


@numba.njit(nogil=True)
def weight_field_old(x, gp, e_j, dx, order=0):
    """Obtain weighted field on particle from the grid.

    Weighting function order determined by value of :order:. Boundary is assumed
    to be periodic such that j = mod(j, m).

    :order: = 0: Nearest-grid-point weighting. Force on particle is given by the
    value of the field at the nearest grid point.

    :order: = 1: Linear weighting. Force on a particle is given by linear
    interpolation between the two nearest grid points.
    """
    e_i = np.zeros_like(x)
    m = gp.shape[0]

    for i, x_i in enumerate(x):
        # Nearest grid point
        if order == 0:
            j = round(x_i / dx)
            e_i[i] = e_j[j % m]

        # Linear
        elif order == 1:
            j = math.floor(x_i / dx)  # 0 <= j <= m
            # Apply periodic boundary conditions
            if j == m:
                e_i[i] = e_j[0]
            elif j == m - 1:
                e_i[i] = (
                    (gp[j] + dx - x_i) * e_j[j] + (x_i - gp[j]) * e_j[0]
                ) / dx
            else:
                e_i[i] = (
                    (gp[j + 1] - x_i) * e_j[j] + (x_i - gp[j]) * e_j[j + 1]
                ) / dx

        else:
            raise ValueError("Incorrect value 'order', must be 0 or 1.")

    return e_i


if __name__ == "__main__":
    print("Testing performance of weighting.py")
    print("#" * 80)
    print("Testing performance of weight_particles(order=0)")
    print("#" * 80)
    for test_m in [32, 1024, 8096]:
        for test_n in [64, 512, 4096]:
            for order in [0, 1]:
                m = test_m
                n = test_n
                t_steps = 100
                print(f"m: {m}, n: {n}, order: {order}")
                grid_pts = np.linspace(0, 1, m + 1)[:-1]
                dx = 1 / m
                x = np.random.uniform(0.0, 1.0, n)
                # Run once to compile weight_particles()
                rho = weight_particles(x, grid_pts, dx, m, order=order)

                ptime_numba = 0
                bar = progressbar.ProgressBar(
                    maxval=t_steps,
                    widgets=[
                        progressbar.Bar("=", "[", "]"),
                        " ",
                        progressbar.Percentage(),
                    ],
                )
                bar.start()
                for step in range(t_steps):
                    x = np.random.uniform(0.0, 1.0, n)
                    start_time = time.perf_counter()
                    rho = weight_particles(x, grid_pts, dx, m, order=order)
                    end_time = time.perf_counter()
                    ptime_numba += end_time - start_time
                    bar.update(step + 1)
                bar.finish()

                ptime_python = 0
                bar = progressbar.ProgressBar(
                    maxval=t_steps,
                    widgets=[
                        progressbar.Bar("=", "[", "]"),
                        " ",
                        progressbar.Percentage(),
                    ],
                )
                bar.start()
                for step in range(t_steps):
                    x = np.random.uniform(0.0, 1.0, n)
                    start_time = time.perf_counter()
                    weight_particles.py_func(x, grid_pts, dx, m, order=order)
                    end_time = time.perf_counter()
                    ptime_python += end_time - start_time
                    bar.update(step + 1)
                bar.finish()

                print(
                    f"(numba ) Total elapsed time per step (n={n}):"
                    f" {10**6 * ptime_numba / t_steps:.3f} µs"
                )
                print(
                    f"(python) Total elapsed time per step (n={n}):"
                    f" {10**6 * ptime_python / t_steps:.3f} µs"
                )
                print(
                    f"numba speedup: {(ptime_python) / (ptime_numba):.2f} times"
                    " faster"
                )
