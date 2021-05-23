import numpy as np

from pic1.model import rotate_xy


def test_rotate_xy():
    theta = np.pi / 2
    test_x = np.array([0, 1, 0.5])
    test_y = np.array([1, 0, 0.5])
    expect_x = np.array([-1, 0, -0.5])
    expect_y = np.array([0, 1, 0.5])
    (new_x, new_y) = rotate_xy(test_x, test_y, theta)
    eps = 10 ** -8
    assert np.linalg.norm(new_x - expect_x) < eps
    assert np.linalg.norm(new_y - expect_y) < eps

    theta = -np.pi / 3
    test_x = np.array([0, 1])
    test_y = np.array([1, 0])
    expect_x = np.array([3 ** (1 / 2) / 2, 1 / 2])
    expect_y = np.array([1 / 2, -(3 ** (1 / 2)) / 2])
    (new_x, new_y) = rotate_xy(test_x, test_y, theta)
    eps = 10 ** -8
    assert np.linalg.norm(new_x - expect_x) < eps
    assert np.linalg.norm(new_y - expect_y) < eps