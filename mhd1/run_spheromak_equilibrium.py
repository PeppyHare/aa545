"""Run Ideal MHD Solver."""
import math
import itertools
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.special as sp
import tables

from mhd1.configuration import CylindricalConfiguration
from mhd1.models import LinearMHDModel
from mhd1.utils import save_data, load_data
import mhd1.plots as plots


# plt.style.use("dark_background")  # For comfortable nighttime coding
mpl.rcParams["image.aspect"] = "auto"
mpl.rcParams["image.cmap"] = "BrBG"
demo_mode = True


class SpheromakConfiguration(CylindricalConfiguration):
    """Linearized m=1 Spheromak equilibrium."""

    dt = 0.001
    t_max = 0.5
    # Number of grid points in r
    Mr = 50

    # Number of grid points in z
    Mz = 50

    max_history_steps = 30

    def __init__(self, R=1.0, L=1.0):
        self.R = R
        self.L = L
        CylindricalConfiguration.__init__(self)

    def set_initial_conditions(self):

        self.initial_v = np.zeros((3, self.Mr, self.Mz), dtype="cfloat")
        self.initial_b = np.zeros((3, self.Mr, self.Mz), dtype="cfloat")
        self.initial_p = np.zeros((self.Mr, self.Mz), dtype="cfloat")

        self.b0r = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.b0t = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.b0z = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.db0rdr = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.db0rdz = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.db0tdr = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.db0tdz = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.db0zdr = np.zeros((self.Mr, self.Mz), dtype="cfloat")
        self.db0zdz = np.zeros((self.Mr, self.Mz), dtype="cfloat")

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
                self.initial_v[0, j, k] += (
                    0.0001 * np.sin(np.pi * r / self.R) * np.cos(kz * z)
                )
        self.p0 = np.zeros((self.Mr, self.Mz), dtype="cfloat") * 0.01
        self.rho0 = np.ones((self.Mr, self.Mz), dtype="cfloat")

        # Normalization
        self.scale_b = max(
            [np.max(self.b0z), np.max(self.b0r), np.max(self.b0t)]
        )
        self.scale_p = np.max(self.p0)


c = SpheromakConfiguration(R=1.0, L=1.0)
m = LinearMHDModel(c)
plots.plot_initial_cylindrical_mhd(m)
m.run()
save_data(m, "mhd_spheromak_test.p")

# m = load_data(
#     os.path.join(
#         "saved_data", "mhd1", "2021-05-28_2546.171462_mhd_spheromak_test.p"
#     )
# )


d = m.d
plt.figure(figsize=(6, 8))
plt.plot(m.d.KE)
plt.xlabel("t")
plt.ylabel("Kinetic energy")
plt.yscale("log")
# plt.show()

plots.animate_cylindrical_mhd(m, add_equilibrium=False)
plots.snapshot_cylindrical_mhd(m, n=0)
plots.snapshot_cylindrical_mhd(m, n=1)
plots.snapshot_cylindrical_mhd(m, n=3)
plots.snapshot_cylindrical_mhd(m, n=5)
plots.snapshot_cylindrical_mhd(m, n=20)
