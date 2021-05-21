"""Run Ideal MHD Solver."""
import math
import itertools
import datetime
from pickle import load

import numpy as np
from matplotlib import pyplot as plt
import tables

from mhd1.configuration import Configuration
from mhd1.models import MHDModel
from mhd1.methods import divB
from mhd1.utils import save_data, load_data
import mhd1.plots as plots


plt.style.use("dark_background")


class ScrewPinchConfiguration(Configuration):
    """Cylindrical screw pinch.

    Parabolic axial current distribution and uniform axial magnetic field.
    """

    max_history_steps = 16

    def __init__(self, Mx=33, My=33, Mz=33, dt=0.005, t_max=15, j0=1.0, R=8):
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

        # Left-hand boundary of domain
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0

        # Right-hand boundary of domain
        self.x_max = Mx + 1
        self.y_max = My + 1
        self.z_max = Mz + 1

        # Time step
        self.dt = dt
        self.t_max = t_max

        # Maximum current density
        self.j0 = j0

        # Pinch radius
        self.R = R

        # Conducting wall boundary conditions
        self.bcx = 1
        self.bcy = 1

        # Periodic boundary conditions
        self.bcz = 0
        Configuration.__init__(self)

    def set_initial_conditions(self):
        """Set the initial conditions to be evolved in time.

        The conserved variables q are stored at each grid point (i, j, k):

        Q[0] = rho
        Q[1] = rho*vx
        Q[2] = rho*vy
        Q[3] = rho*vz
        Q[4] = Bx
        Q[5] = By
        Q[6] = Bz
        Q[7] = e

        Initial configuration:
        """

        Q = np.zeros((8, self.Mx, self.My, self.Mz))
        mu = self.mu
        gamma = self.gamma

        # # Constant background density
        # rho0 = 0.05
        # Q[0, :, :, :] += rho0
        # Constant temperature
        T = 1
        # Ion mass
        mi = 1
        # Current profile: J0(1-r^2/R^2)
        # Pinch radius:
        # R = (self.Mx - 1) / 4
        R = self.R
        # Peak current
        j0 = 2
        # Constant background pressure
        pmax = 5 * j0 ** 2 * R ** 2 * mu / 48
        p0 = 0.5 * pmax
        # Constant B_z
        Q[6, :, :, :] += 1
        # Origin in x-y plane
        io = int((self.Mx - 1) / 2)
        jo = int((self.My - 1) / 2)
        p_mat = np.zeros((self.Mx, self.My))
        for i, j in itertools.product(range(self.Mx), range(self.My)):
            x = (i - io) * self.dx
            y = (j - jo) * self.dy
            r = math.sqrt(x ** 2 + y ** 2)
            theta = math.atan2(y, x)
            if r <= R:
                # Q[3, i, j, :] += J0 * (1 - r ** 2 / R ** 2) * Q[0, i, j, :]
                btheta = j0 * mu / 2 * (r - r ** 3 / (2 * R ** 2))
                p = (
                    p0
                    + 5 * j0 ** 2 * R ** 2 * mu / 48
                    - j0 ** 2
                    * mu
                    * r ** 2
                    * (2 * r ** 4 - 9 * r ** 2 * R ** 2 + 12 * R ** 4)
                    / (48 * R ** 4)
                )
                p_mat[i, j] += p
                Q[4, i, j, :] -= math.sin(theta) * btheta
                Q[5, i, j, :] += math.cos(theta) * btheta
            else:
                btheta = j0 * mu * R ** 2 / 4 / r
                p = p0
                p_mat[i, j] += p
                Q[4, i, j, :] -= math.sin(theta) * btheta
                Q[5, i, j, :] += math.cos(theta) * btheta
            # Assume isothermal ideal gas
            Q[0, i, j, :] += p * mi / T
        e = (
            p / (gamma - 1)
            + (Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2) / (2 * Q[0])
            + (Q[4] ** 2 + Q[5] ** 2 + Q[6] ** 2) / (2 * mu)
        )
        Q[7] += e
        print(f"Pressure along x-axis: {p_mat[:, jo]}")

        # # Plot the plasma pressure profile
        # plt.figure()
        # plt.plot(p_mat[:, jo])
        # plt.xlabel(r"x")
        # plt.ylabel(r"p")
        # plt.title("Plasma Pressure")
        # plt.show()

        # dB = divB(
        #     B=Q[4:7],
        #     dx=self.dx,
        #     dy=self.dy,
        #     dz=self.dz,
        #     bcx=self.bcx,
        #     bcy=self.bcy,
        #     bcz=self.bcz,
        # )
        # plt.figure()
        # plt.imshow(dB[:, :, int(self.Mz / 2)], origin="lower")
        # plt.colorbar()
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$y$")
        # plt.title(r"$\nabla \cdot B$")
        # plt.show()
        self.initial_Q = Q


c = ScrewPinchConfiguration(Mx=65, My=65, Mz=65, dt=0.01, t_max=1, j0=0.01, R=4)
m = MHDModel(c, check_divB=True)

plots.plot_initial_distribution_xy(m)
m.run()
save_data(m, f"screw_pinch_latest.p")

m = load_data("saved_data/mhd1/2021-05-20_59772.789067_screw_pinch_latest.p")

plots.animate_mhd(m)

# fig2 = plt.figure()
# ax_divb = fig2.add_subplot(111)
# ax_divb.plot(m.d.max_divB)
# plt.xlabel("$t$")
# plt.ylabel(r"max($\nabla \cdot B$)")

# fig1 = plt.figure(figsize=(6, 8))
# ax_e = fig1.add_subplot(111)
# ax_e.plot(m.d.KE, label="KE")
# ax_e.plot(m.d.TE, label="TE")
# ax_e.plot(m.d.FE, label="FE")
# plt.legend()
# plt.xlabel(r"$t$")
# plt.ylabel(r"Energy")
# plt.yscale("log")

# plt.show()

# f = tables.open_file(m.d.h5_filename)
