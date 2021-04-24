import math

import numpy as np
import numba


@numba.njit
def weight_particles(x, gp, dx, M, q=1, order=0):
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
        for x_i in x:
            j = round(x_i / dx)
            rho[j % M] += q / dx

    # Linear weighting
    elif order == 1:
        for x_i in x:
            j = math.floor(x_i / dx)  # 0 <= j <= m
            # Apply periodic boundary conditions
            if j == M:
                rho[0] += q / dx
            elif j == M - 1:
                rho[j] += q * (gp[j] + dx - x_i) / dx ** 2
                rho[0] += q * (x_i - gp[j]) / dx ** 2
            else:
                rho[j] += q * (gp[j + 1] - x_i) / dx ** 2
                rho[j + 1] += q * (x_i - gp[j]) / dx ** 2
    else:
        raise ValueError("Incorrect value 'order', must be 0 or 1.")
    assert np.sum(rho[:-1] * dx) - x.size < 10 ** -6  # charge conservation
    return rho


@numba.njit
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
