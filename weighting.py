import math

import numpy as np
import numba


@numba.njit(boundscheck=True)
def weight_particles(x, gp, dx, m, order=0):
    """Weight particles to grid.

    Weighting function order determined by value of :order:

    :order: = 0: Nearest-grid-point weighting. Boundary is assumed to be
    periodic, so particles assigned to the left-most grid point are also
    assigned to the right-most, and vice-versa.

    :order: = 1: Linear weighting. Particles are assigned to the two nearest
    grid points via linear interpolation.
    """

    rho = np.zeros_like(gp)

    # Nearest grid point
    if order == 0:
        for x_i in x:
            j = int(x_i / dx)
            rho[j] += 1 / dx
            # Apply periodic boundary conditions
            if j == 0:
                rho[m - 1] += 1 / dx
            if j == m - 1:
                rho[0] += 1 / dx

    # Linear weighting
    elif order == 1:
        for x_i in x:
            j = math.floor(x_i / dx)  # 0 <= j <= m - 1
            # Apply periodic boundary conditions
            if j == m - 1:
                j = 0
            rho[j] += (gp[j + 1] - x_i) / dx ** 2
            rho[j + 1] += (x_i - gp[j]) / dx ** 2
            if j == 0:
                rho[m - 1] += (gp[j + 1] - x_i) / dx ** 2
            if j == m - 2:
                rho[0] += (x_i - gp[j]) / dx ** 2
    else:
        raise ValueError("Incorrect value 'order', must be 0 or 1.")
    assert np.sum(rho[:-1] * dx) - x.size < 10 ** -6  # charge conservation
    return rho


@numba.njit(boundscheck=True)
def weight_field(x, gp, e_j, dx, order=0):
    """Obtain weighted field on particle from the grid.

    Weighting function order determined by value of :order:

    :order: = 0: Nearest-grid-point weighting. Force on particle is given by the
    value of the field at the nearest grid point.

    :order: = 1: Linear weighting. Force on a particle is given by linear
    interpolation between the two nearest grid points.

    Assumes periodic boundary conditions (e_j[0] == e_j[-1])
    """
    e_i = np.zeros_like(x)
    m = gp.shape[0]

    for i, x_i in enumerate(x):
        # Nearest grid point
        if order == 0:
            j = int(x_i / dx)
            e_i[i] = e_j[j]

        # Linear
        elif order == 1:
            j = math.floor(x_i / dx)  # 0 <= j <= m - 1
            # Apply periodic boundary conditions
            if j == m - 1:
                j = 0
            e_i[i] = (
                (gp[j + 1] - x_i) * e_j[j] + (x_i - gp[j]) * e_j[j + 1]
            ) / dx

        else:
            raise ValueError("Incorrect value 'order', must be 0 or 1.")

    return e_i
