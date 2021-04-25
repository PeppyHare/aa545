"""Poisson solvers."""
import time

import numpy as np
import numba
import progressbar

from weighting import weight_particles


def setup_poisson(m):
    """Generate 2nd order finite difference matrix for Poisson equation.

    There is a gauge ambiguity inherent to the boundary condition. There are a
    couple of ways we can set the gauge. The method used here is to fix the
    value u_0 = p_0

    The form of A is:

    [1   0   0  ...  0   0   0][u_0  ] = [p_0  ]
    [1  -2   1  ...  0   0   0][u_1  ] = [p_1  ]
    [0   1  -2  ...  0   0   0][u_2  ] = [p_2  ]
    [           ...           ][u_j  ] = [p_j  ]
    [0   0   0  ... -2   1   0][u_m-3] = [p_m-3]
    [0   0   0  ...  1  -2   1][u_m-2] = [p_m-2]
    [1   0   0  ...  0   1  -2][u_m-1] = [p_m-1]
    """
    a = np.zeros((m, m))
    a[0, 0] = 1
    for j in range(1, m - 1):
        a[j, j - 1] = -1
        a[j, j] = 2
        a[j, j + 1] = -1
    a[m - 1, 0] = -1
    a[m - 1, m - 1] = 2
    a[m - 1, m - 2] = -1
    return (np.linalg.inv(a), a)


def setup_poisson2(m):
    """Generate 2nd order finite difference matrix for Poisson equation.

    There is a gauge ambiguity inherent to the boundary condition. There are a
    couple of ways we can set the gauge. The method used here is to fix the
    value u_0 = 0

    The form of A is:


    [-2   1  ...  0   0   0][u_1  ]   [p_1  ]
    [ 1  -2  ...  0   0   0][u_2  ]   [p_2  ]
    [        ...           ][u_j  ] = [p_j  ]
    [ 0   0  ... -2   1   0][u_m-3]   [p_m-3]
    [ 0   0  ...  1  -2   1][u_m-2]   [p_m-2]
    [ 0   0  ...  0   1  -2][u_m-1]   [p_m-1]
    """
    a = np.zeros((m - 1, m - 1))
    a[0, 0] = -2
    a[0, 1] = 1
    for j in range(1, m - 2):
        a[j, j - 1] = 1
        a[j, j] = -2
        a[j, j + 1] = 1
    a[m - 2, m - 2] = -2
    a[m - 2, m - 3] = 1
    return (np.linalg.inv(a), a)


@numba.njit
def solve_poisson(rho, inv_a, dx):
    """Calculate electric field on the grid.

    :rho: is the weighted charge density at all grid points.
    :inv_a: is the pre-computed second order finite difference matrix for the
    Poisson equation with periodic boundary conditions and gauge such that
    phi[0] = 0.
    :dx: is the grid spacing.
    """
    return np.dot(inv_a, rho) * dx ** 2


@numba.njit
def solve_poisson2(rho, inv_a, dx):
    """Calculate electric field on the grid.

    :rho: is the weighted charge density at all grid points.
    :inv_a: is the pre-computed second order finite difference matrix for the
    Poisson equation with periodic boundary conditions and gauge such that
    phi[0] = 0.
    :dx: is the grid spacing.
    """
    phi = np.zeros_like(rho)
    phi[1:] = -1 * np.dot(inv_a, rho[1:]) * dx ** 2
    return phi


@numba.njit
def compute_field(rho, inv_a, dx):
    """Differentiate phi to get electric field at grid points."""
    rho[0] = 0
    phi = solve_poisson(rho, inv_a, dx)
    e_j = np.zeros_like(phi)
    # First-order centered finite difference
    for j in numba.prange(1, phi.shape[0] - 1):
        e_j[j] = (phi[j - 1] - phi[j + 1]) / (2 * dx)
    e_j[0] = (phi[-1] - phi[1]) / (2 * dx)
    e_j[-1] = (phi[-2] - phi[0]) / (2 * dx)
    return e_j


@numba.njit
def compute_field2(rho, inv_a, dx):
    """Differentiate phi to get electric field at grid points."""
    phi = solve_poisson2(rho, inv_a, dx)
    e_j = np.zeros_like(phi)
    # First-order centered finite difference
    for j in numba.prange(1, phi.shape[0] - 1):
        e_j[j] = (phi[j - 1] - phi[j + 1]) / (2 * dx)
    e_j[0] = (phi[-1] - phi[1]) / (2 * dx)
    e_j[-1] = (phi[-2] - phi[0]) / (2 * dx)
    return e_j


if __name__ == "__main__":
    print("#" * 80)
    print("Testing solution of uniform charge distribution")
    print("#" * 80)
    m = 33
    n = 4
    x = np.linspace(0.0, 1.0, n)
    grid_pts = np.linspace(0, 1, m + 1)[:-1]
    dx = 1 / (m - 1)
    rho = weight_particles(x, grid_pts, dx, m, order=1)
    (inv_a, _) = setup_poisson(m)
    print(inv_a)
    e_j = compute_field(rho, inv_a, dx)
    print(e_j)

    print("#" * 80)
    print("Testing performance of solve_poisson()")
    print("#" * 80)
    m = 32
    n = 4098
    t_steps = 10000
    grid_pts = np.linspace(0, 1, m)
    dx = 1 / (m - 1)
    x = np.random.uniform(0.0, 1.0, n)
    # Run one short iteration to compile compute_field()
    rho = weight_particles(x, grid_pts, dx, m, order=1)
    (inv_a, _) = setup_poisson(m)
    compute_field(rho, inv_a, dx)

    ptime_numba = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x = np.random.uniform(0.0, 1.0, n)
        rho = weight_particles(x, grid_pts, dx, m, order=1)
        start_time = time.perf_counter()
        compute_field(rho, inv_a, dx)
        end_time = time.perf_counter()
        ptime_numba += end_time - start_time
        bar.update(step + 1)
    bar.finish()

    ptime_python = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x = np.random.uniform(0.0, 1.0, n)
        weight_particles(x, grid_pts, dx, m, order=1)
        start_time = time.perf_counter()
        compute_field.py_func(rho, inv_a, dx)
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
    print(f"numba speedup: {(ptime_python) / (ptime_numba):.2f} times faster")
