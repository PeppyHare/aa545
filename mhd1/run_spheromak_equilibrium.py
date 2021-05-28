"""Run Ideal MHD Solver."""
import math
import itertools
import os

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sp
import tables

from mhd1.configuration import CylindricalConfiguration
from mhd1.models import LinearMHDModel
from mhd1.utils import save_data, load_data
import mhd1.plots as plots


plt.style.use("dark_background")  # For comfortable nighttime coding
demo_mode = True


class SpheromakConfiguration(CylindricalConfiguration):
    """Linearized m=1 Spheromak equilibrium."""

    dt = 0.001
    t_max = 5
    # Number of grid points in r
    Mr = 11

    # Number of grid points in z
    Mz = 11

    def __init__(self, R=1.0, L=1.0):
        self.R = R
        self.L = L
        CylindricalConfiguration.__init__(self)

    def set_initial_conditions(self):

        self.initial_v = np.zeros((3, self.Mr, self.Mz), dtype="cfloat")
        self.initial_b = np.zeros((3, self.Mr, self.Mz), dtype="cfloat")
        self.initial_p = np.zeros((self.Mr, self.Mz), dtype="cfloat")

        self.b0r = np.zeros((self.Mr, self.Mz))
        self.b0t = np.zeros((self.Mr, self.Mz))
        self.b0z = np.zeros((self.Mr, self.Mz))
        self.db0rdr = np.zeros((self.Mr, self.Mz))
        self.db0rdz = np.zeros((self.Mr, self.Mz))
        self.db0tdr = np.zeros((self.Mr, self.Mz))
        self.db0tdz = np.zeros((self.Mr, self.Mz))
        self.db0zdr = np.zeros((self.Mr, self.Mz))
        self.db0zdz = np.zeros((self.Mr, self.Mz))

        kr = 3.832 / self.R
        kz = np.pi / self.L
        l0 = math.sqrt(kr ** 2 + kz ** 2)

        for j in range(self.Mr):
            for k in range(self.Mz):
                r = self.dr * j
                z = self.dz * k
                self.b0r[j, k] += -kz * sp.jv(1, kr * r) * np.cos(kz * z)
                self.db0rdr[j, k] += (
                    -kz * kr * sp.jvp(1, kr * r, 1) * np.cos(kz * z)
                )
                self.db0rdz[j, k] += kz * kz * sp.jv(1, kr * r) * np.sin(kz * z)
                self.b0t[j, k] += l0 * sp.jv(1, kr * r) * np.sin(kz * z)
                self.db0tdr[j, k] += (
                    l0 * kr * sp.jvp(1, kr * r, 1) * np.sin(kz * z)
                )
                self.db0tdz[j, k] += l0 * kz * sp.jv(1, kr * r) * np.cos(kz * z)
                self.b0z[j, k] += kr * sp.jv(0, kr * r) * np.sin(kz * z)
                self.db0zdr[j, k] += (
                    kr * kr * sp.jvp(0, kr * r, 1) * np.sin(kz * z)
                )
                self.db0zdz[j, k] += kr * kz * sp.jv(0, kr * r) * np.cos(kz * z)
                self.initial_v[:, j, k] += (
                    0.01 * np.sin(kr * r) * np.sin(kz * z)
                )

        self.p0 = np.ones((self.Mr, self.Mz)) * 0.01
        self.rho0 = np.ones((self.Mr, self.Mz))


c = SpheromakConfiguration(R=1.0, L=2.0)
m = LinearMHDModel(c)
m.run()
save_data(m, "mhd_spheromak_test.p")

# m = load_data(
#     os.path.join(
#         "saved_data", "mhd1", "2021-05-26_52194.270265_mhd_spheromak_test.p"
#     )
# )

d = m.d
print(f"v:{d.v}")
plt.figure()
with tables.open_file(m.d.h5_filename, "r") as f:
    v_hist = f.root.v
    v_norm = (
        np.real(v_hist[:, 0, :, :]) ** 2
        + np.real(v_hist[:, 1, :, :]) ** 2
        + np.real(v_hist[:, 2, :, :]) ** 2
    )
    plt.plot(np.sum(v_norm, axis=(1, 2)))
    plt.yscale("log")
    plt.show()

# plots.mhd_snapshot(m, n=5)
# plots.shock_snapshots(m, n=0)
# plots.plot_shock(m)


# if demo_mode:
#     m = load_data(
#         os.path.join(
#             "saved_data", "mhd1", "2021-05-23_83581.340669_screw_pinch_latest.p"
#         )
#     )
# else:
#     c = ScrewPinchConfiguration(
#         Mx=65, My=65, Mz=65, dt=0.05, t_max=10, j0=0.5, R=8
#     )
#     m = MHDModel(c, check_divB=True)
#     plots.plot_initial_distribution_all_axes(m)

#     m.run()
#     save_data(m, "screw_pinch_latest.p")


# plots.mhd_snapshot(m, 0)
# plots.animate_mhd(m)
