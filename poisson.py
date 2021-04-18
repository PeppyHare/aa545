import time

import numpy as np
import numba
import progressbar

from weighting import weight_particles


def setup_poisson(m):
    """Generate 2nd order finite difference matrix for Poisson equation.

    To set the gauge and provide the second boundary condition, we constrain
    u[0] = 0. The form of A is:

    [1   0   0  ...  0   0   0]
    [1  -2   1  ...  0   0   0]
    [0   1  -2  ...  0   0   0]
    [           ...           ]
    [0   0   0  ... -2   1   0]
    [0   0   0  ...  1  -2   1]
    [1   0   0  ...  0   1  -2]
    """
    a = np.zeros((m - 1, m - 1))
    a[0, 0] = 1
    for j in range(1, m - 2):
        a[j, j - 1] = 1
        a[j, j] = -2
        a[j, j + 1] = 1
    a[m - 2, 0] = 1
    a[m - 2, m - 3] = 1
    a[m - 2, m - 2] = -2
    return np.linalg.inv(a)


@numba.njit(boundscheck=True)
def solve_poisson(rho, inv_a, dx):
    """Calculate electric field on the grid.

    :rho: is the weighted charge density at all grid points.
    :inv_a: is the pre-computed second order finite difference matrix for the
    Poisson equation with periodic boundary conditions and gauge such that
    phi[0] = 0.
    :dx: is the grid spacing.
    """
    phi = np.dot(inv_a, rho[:-1])
    e_j = np.zeros_like(rho)

    # First-order centered finite difference
    for j in numba.prange(1, rho.size - 2):
        e_j[j] = (phi[j - 1] - phi[j + 1]) / 2 * dx
    e_j[0] = (phi[-1] - phi[1]) / 2 * dx
    e_j[-1] = e_j[0]
    return e_j


if __name__ == "__main__":
    print("#" * 80)
    print("Testing solution of uniform charge distribution")
    print("#" * 80)
    m = 33
    n = 4
    x = np.linspace(0.0, 1.0, n)
    grid_pts = np.linspace(0, 1, m)
    dx = 1 / (m - 1)
    rho = weight_particles(x, grid_pts, dx, m, order=1)
    inv_a = setup_poisson(m)
    print(inv_a)
    e_j = solve_poisson(rho, inv_a, dx)
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
    # Run one short iteration to compile solve_poisson()
    rho = weight_particles(x, grid_pts, dx, m, order=1)
    inv_a = setup_poisson(m)
    solve_poisson(rho, inv_a, dx)

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
        solve_poisson(rho, inv_a, dx)
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
        solve_poisson.py_func(rho, inv_a, dx)
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
