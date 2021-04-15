import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def weight_nearest(pos, gp, n, m, dx):
    rho = np.zeros_like(gp)
    # for i in numba.prange(np.size(pos)):
    # do thing
    return rho


@numba.jit(nopython=True, parallel=True)
def weight_linear(pos, gp, n, m, dx):
    rho = np.zeros_like(gp)
    # for i in numba.prange(np.size(pos)):
    # do thing
    return rho
