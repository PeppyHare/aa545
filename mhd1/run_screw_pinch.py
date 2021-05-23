"""Run Ideal MHD Solver."""
import math
import itertools

import numpy as np
from matplotlib import pyplot as plt

from mhd1.configuration import Configuration
from mhd1.models import MHDModel
from mhd1.methods import divB
from mhd1.utils import save_data, load_data
import mhd1.plots as plots


plt.style.use("dark_background")  # For comfortable nighttime coding


class RiemannShockConfiguration(Configuration):
    """Brio-Wu MHD shock tube.

    MHD variant of Sod's shock tube problem.
    See https://doi.org/10.1016/0021-9991(88)90120-9
    """

    max_history_steps = 40

    def __init__(self):
        self.Mx = 800
        self.My = 3
        self.Mz = 3

        self.x_min = 0
        self.y_min = 0
        self.z_min = 0

        self.x_max = self.Mx - 1
        self.y_max = self.My - 1
        self.z_max = self.Mz - 1

        self.dt = 0.02
        self.t_max = 400
        Configuration.__init__(self)

    def set_initial_conditions(self):
        Q = np.zeros((8, self.Mx, self.My, self.Mz))
        x2 = int(self.Mx / 2)
        gamma = 2
        mu = 1

        # Bx = 0.75 everywhere
        Bx = 0.75
        Q[4, :, :, :] = Bx

        # Left state
        rho_l = 1
        p_l = 1
        By_l = 1
        Q[0, :x2, :, :] = rho_l
        Q[5, :x2, :, :] = By_l
        Q[7, :x2, :, :] = p_l / (gamma - 1) + (Bx ** 2 + By_l ** 2) / (2 * mu)

        # Right state
        rho_r = 0.125
        p_r = 0.1
        By_r = -1
        Q[0, x2:, :, :] = rho_r
        Q[5, x2:, :, :] = By_r
        Q[7, x2:, :, :] = p_r / (gamma - 1) + (Bx ** 2 + By_r ** 2) / (2 * mu)
        self.initial_Q = Q


class ScrewPinchConfiguration(Configuration):
    """Cylindrical screw pinch.

    Parabolic axial current distribution and uniform axial magnetic field.
    """

    max_history_steps = 64

    def __init__(
        self, Mx=33, My=33, Mz=33, dt=0.005, t_max=15, j0=1.0, R=8, B0=1
    ):
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

        # Left-hand boundary of domain
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0

        # Right-hand boundary of domain
        self.x_max = Mx - 1
        self.y_max = My - 1
        self.z_max = Mz - 1

        # Time step
        self.dt = dt
        self.t_max = t_max

        # Maximum current density
        self.j0 = j0

        # Axial magnetic field
        self.B0 = B0

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

        # Constant temperature
        T = 1
        # Ion mass
        mi = 1
        # Current profile: J0(1-r^2/R^2)
        # Pinch radius:
        # R = (self.Mx - 1) / 4
        R = self.R
        # Peak current
        j0 = self.j0
        # Constant background pressure
        pmax = 5 * j0 ** 2 * R ** 2 * mu / 48
        p0 = 0.1 * pmax
        # Constant B_z
        Q[6, :, :, :] += self.B0
        # Origin in x-y plane
        io = int((self.Mx - 1) / 2)
        jo = int((self.My - 1) / 2)
        p_mat = np.zeros((self.Mx, self.My))
        btheta_mat = np.zeros((self.Mx, self.My))
        for i, j in itertools.product(range(self.Mx), range(self.My)):
            x = (i - io) * self.dx
            y = (j - jo) * self.dy
            r = math.sqrt(x ** 2 + y ** 2)
            theta = math.atan2(y, x)
            if r <= R:
                # Q[3, i, j, :] += J0 * (1 - r ** 2 / R ** 2) * Q[0, i, j, :]
                btheta = j0 * mu / 2 * (r - r ** 3 / (2 * R ** 2))
                btheta_mat[i, j] += btheta
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
                btheta = j0 * mu * R ** 2 / (4 * r)
                btheta_mat[i, j] += btheta
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

        # Plot the plasma pressure profile
        plt.figure()
        plt.plot(p_mat[:, jo], label=r"$p$")
        plt.plot(btheta_mat[:, jo], label=r"$B_ \theta$")
        plt.xlabel(r"x")
        plt.title("Initial Pressure Profile")
        plt.legend()
        plt.show()

        dB = divB(
            B=Q[4:7],
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            bcx=self.bcx,
            bcy=self.bcy,
            bcz=self.bcz,
        )
        plt.figure()
        plt.imshow(dB[:, :, int(self.Mz / 2)], origin="lower")
        plt.colorbar()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title(r"$\nabla \cdot B$")
        plt.show()
        self.initial_Q = Q


c = ScrewPinchConfiguration(Mx=65, My=65, Mz=65, dt=0.05, t_max=10, j0=0.5, R=8)
m = MHDModel(c, check_divB=True)
plots.plot_initial_distribution_all_axes(m)

m.run()
save_data(m, "screw_pinch_latest.p")

# m = load_data("saved_data/mhd1/2021-05-21_82451.590781_screw_pinch_latest.p")

plots.animate_mhd(m)
plots.mhd_snapshot(m, 3)


c = RiemannShockConfiguration()
m = MHDModel(c, check_divB=False)
m.run()
save_data(m, "riemann_shock.p")

# m = load_data("saved_data/mhd1/2021-05-21_76895.019152_riemann_shock.p")

plots.shock_snapshots(m, n=0)
plots.shock_snapshots(m, n=2)
plots.shock_snapshots(m, n=5)

plots.plot_shock(m)
