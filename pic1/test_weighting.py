import numpy as np

from pic1.weighting import weight_particles, weight_field


def test_ngp_weight_particles():
    """Nearest grid point particle weighting.

    Spatial periodic domain: [0, 1)
    Test grid: [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Test points: [0.0, 0.1, 0.45, 0.54, 0.89, 0.99]
    Expected : [20.0, 10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 10.0]
    """
    eps = 10 ** -6
    M = 10
    grid_pts = np.linspace(0, 1, M + 1)[:-1]
    dx = 1 / M
    x = np.array([0.0, 0.1, 0.45 + eps, 0.54, 0.89, 0.99])
    expected = np.array([20.0, 10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 10.0])
    rho = weight_particles(x, grid_pts, dx, M, q=1, order=0)
    assert np.linalg.norm(expected - rho) < eps


def test_linear_weight_particles():
    """Linear interpolation particle weighting.

    Spatial periodic domain: [0, 1)
    Test grid: [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Test points: [0.0, 0.1, 0.45, 0.54, 0.89, 0.99, 1.0]
    Expected : [29.0, 10.0, 0.0, 0.0, 5.0, 11.0, 4.0, 0.0, 0.1, 10.0]
    """
    eps = 10 ** -6
    M = 10
    grid_pts = np.linspace(0, 1, M + 1)[:-1]
    dx = 1 / M
    x = np.array([0.0, 0.1, 0.45, 0.54, 0.89, 0.99, 1.0])
    expected = np.array([29.0, 10.0, 0.0, 0.0, 5.0, 11.0, 4.0, 0.0, 1.0, 10.0])
    rho = weight_particles(x, grid_pts, dx, M, q=1, order=1)
    assert np.linalg.norm(expected - rho) < eps


def test_linear_weight_particles2():
    """Linear interpolation particle weighting. This time with more edge cases

    Spatial periodic domain: [0, 1)
    Test grid: [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    Test points: [0.0, 0.1, 0.45, 0.5, 0.54, 0.69, 0.79, 0.8]
    Expected : [23.2, 10.0, 0.0, 0.0, 5.0, 21.0, 5.0, 10.0]
    """
    eps = 10 ** -6
    M = 8
    grid_pts = np.linspace(0, 1, M + 1)[:-1]
    dx = 1 / M
    print(grid_pts)
    x = np.array([0.0, 0.1, 0.45, 0.5, 0.54, 0.69, 0.79, 0.8]) / 0.8
    expected = np.array([23.2, 8, 0, 0, 4, 16.8, 4, 8])
    rho = weight_particles(x, grid_pts, dx, M, q=1, order=1)
    assert np.linalg.norm(expected - rho) < eps


def test_linear_weight_particles3():
    """Linear interpolation particle weighting. This time with a lot of points

    Spatial periodic domain: [0, 1)
    Test grid: [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    Test points: [0.0, 0.1, 0.45, 0.5, 0.54, 0.69, 0.79, 0.8]
    Expected : [23.2, 10.0, 0.0, 0.0, 5.0, 21.0, 5.0, 10.0]
    """
    eps = 10 ** -6
    M = 256
    grid_pts = np.linspace(0, 1, M + 1)[:-1]
    dx = 1 / M
    N = 1024
    x = np.linspace(0, 1, N)
    x += 0.01 * np.sin(np.pi * x)
    rho = weight_particles(x, grid_pts, dx, M, q=1, order=1)
    assert abs(np.sum(rho * dx) - x.size) < eps  # charge conservation


def test_ngp_weight_field():
    """Nearest grid point field weighting.

    Test field: [1.0 , 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
    Test points: [0.0, 0.1, 0.45, 0.54, 0.89, 0.99]
    Expected: [1.0, 0.8, 0, 0, -0.8, 1.0]
    """
    eps = 10 ** -6
    M = 10
    grid_pts = np.linspace(0, 1, M + 1)[:-1]
    dx = 1 / M
    e_j = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8])
    x = np.array([0.0, 0.1, 0.45 + eps, 0.54, 0.89, 0.99])
    expected = np.array([1.0, 0.8, 0, 0, -0.8, 1.0])
    e_i = weight_field(x, grid_pts, e_j, dx, order=0)
    assert np.linalg.norm(expected - e_i) < eps


def test_linear_weight_field():
    """Linear interpolation field weighting.

    Test field: [1.0 , 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
    Test points: [0.0, 0.1, 0.45, 0.54, 0.89, 0.99]
    Expected: [1.0, 0.8, 0, 0, -0.8, 1.0]
    """
    eps = 10 ** -6
    M = 10
    grid_pts = np.linspace(0, 1, M + 1)[:-1]
    dx = 1 / M
    e_j = np.array([10, 8, 6, 4, 2, 0, -2, -4, -6, -8])
    x = np.array([0.0, 0.1, 0.45, 0.54, 0.89, 0.99, 1.0])
    expected = np.array([10, 8, 1, -0.8, -7.8, 8.2, 10])
    e_i = weight_field(x, grid_pts, e_j, dx, order=1)
    assert np.linalg.norm(expected - e_i) < eps
