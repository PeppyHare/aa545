from pic1.poisson import (
    setup_poisson,
    setup_poisson2,
    solve_poisson,
    solve_poisson2,
)
import numpy as np


def test_setup_poisson():
    """Verify that poisson finite difference matrix is invertible, and that it
    is being successfully inverted."""
    eps = 10 ** -6
    for m in range(6, 100):
        inv_a, a = setup_poisson(m)
        assert np.sum(np.dot(inv_a, a) - np.eye(a.shape[0])) < eps, (
            "setup_poisson did not successfully invert the finite difference"
            " matrix."
        )


def test_setup_poisson2():
    """Verify that poisson finite difference matrix is invertible, and that it
    is being successfully inverted."""
    eps = 10 ** -6
    for m in range(6, 100):
        inv_a, a = setup_poisson2(m)
        assert np.sum(np.dot(inv_a, a) - np.eye(a.shape[0])) < eps, (
            "setup_poisson did not successfully invert the finite difference"
            " matrix."
        )


def test_solve_poisson():
    """Validate Poisson finite difference solver.

    Poisson equation to be solved:
    phi''(x) = - rho(x)
    Test function
    rho = sin(2πkx/L)
    Analytic solution to Poisson's equation is:
    phi = (2πk/L)^-2 * sin(2πkx/L)
    """

    for m in [2 ** (2 * x + 1) for x in range(2, 6)]:
        for k in [1, 2, 4, 16]:
            L = 1
            dx = L / m
            (inv_a, _) = setup_poisson(m)
            grid_pts = np.linspace(0, L, m + 1)[:-1]
            test_rho = np.sin(k * 2 * np.pi / L * grid_pts)
            soln = test_rho / ((k * 2 * np.pi / L) ** 2)
            phi = solve_poisson(rho=test_rho, inv_a=inv_a, dx=dx)
            err = np.linalg.norm(soln - phi, ord=1)
            print(f"m={m}, k={k}, err={err}, m*err={m*err}")
            # Each value of phi has error of O(∆x^2 = 1/m^2). We sum over m such
            # values, for a 1-norm error which is O(1/m).
            assert err < L / m


def test_solve_poisson2():
    """Validate Poisson finite difference solver.

    Poisson equation to be solved:
    phi''(x) = - rho(x)
    Test function
    rho = sin(2πkx/L)
    Analytic solution to Poisson's equation is:
    phi = (2πk/L)^-2 * sin(2πkx/L)
    """

    for m in [2 ** (2 * x + 1) for x in range(1, 6)]:
        for k in [1, 2, 4, 16]:
            L = 2 * np.pi
            dx = L / m
            (inv_a, _) = setup_poisson2(m)
            grid_pts = np.linspace(0, L, m + 1)[:-1]
            test_rho = np.sin(k * 2 * np.pi / L * grid_pts)
            soln = test_rho / ((k * 2 * np.pi / L) ** 2)
            phi = solve_poisson2(rho=test_rho, inv_a=inv_a, dx=dx)
            err = np.linalg.norm(soln - phi, ord=1)
            print(f"m={m}, k={k}, err={err}, m*err={m*err}")
            # Each value of phi has error of O(∆x^2 = 1/m^2). We sum over m such
            # values, for a 1-norm error which is O(1/m).
            assert err < L / m
