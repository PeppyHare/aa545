"""Investigate various finite difference methods."""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import math

from configuration import Configuration
from models import AdvectionModel
from methods import (
    exact_advection_dirichlet,
    exact_advection_periodic,
    time_step_ftcs_dirichlet,
    time_step_lax_dirichlet,
    time_step_lax_wendroff_dirichlet,
    time_step_simple_upwind_dirichlet,
    time_step_ftcs_periodic,
)
import plots as plots
from utils import save_plot


# plt.style.use("dark_background")


class LinearAdvectionConfiguration(Configuration):
    """Square pulse."""

    max_history_steps = 500
    time_step_method = staticmethod(exact_advection_dirichlet)

    def set_initial_conditions(self):
        # u = np.sin(np.arange(0, self.M) * 2 * np.pi / self.M)
        u = np.zeros(self.M)
        u[int((self.M - 1) / 20) : int((self.M - 1) / 10 + 1)] = 1
        self.initial_u = u

    def __init__(self, c=1, M=201, dt=0.5):
        self.c = c
        self.M = M
        self.x_min = 0
        self.x_max = M - 1
        self.t_max = 0.9 * abs(M / c)
        self.dt = dt
        Configuration.__init__(self)


class FTCSConfiguration(LinearAdvectionConfiguration):
    time_step_method = staticmethod(time_step_ftcs_dirichlet)


class UpwindConfiguration(LinearAdvectionConfiguration):
    time_step_method = staticmethod(time_step_simple_upwind_dirichlet)


class LaxConfiguration(LinearAdvectionConfiguration):
    time_step_method = staticmethod(time_step_lax_dirichlet)


class LaxWendroffConfiguration(LinearAdvectionConfiguration):
    time_step_method = staticmethod(time_step_lax_wendroff_dirichlet)


# def exact_advection_dirichlet(c: Configuration, t: float):
#     """Shift all spatial indices forward by c*t/dx."""
#     u_soln = np.zeros(c.M)
#     for j in range(c.M):
#         j_new = int(j - c.c * t / c.dx)
#         if j_new <= c.M and j_new > 0:
#             u_soln[j] = c.initial_u[j_new]
#     return u_soln


# c = LinearAdvectionConfiguration()
# m = AdvectionModel(c)
# m.run()
# plt.figure()
# plt.plot(c.x_j, m.d.u_j, label="pure_advection")
# plt.plot(c.x_j, exact_advection_dirichlet(c, c.t_max), label="exact_advection")
# plt.legend()
# a1 = plots.animate_fd(m, plot_title="Exact Solution", repeat=True, hold=False)

# c = FTCSConfiguration()
# m2 = AdvectionModel(c)
# m2.run()
# plots.plot_initial_distribution(m2)
# plots.plot_snapshots(m2, plot_title="FTCS Method", hold=False)
# a2 = plots.animate_fd(m2, plot_title="FTCS Method", repeat=True, hold=False)
# plt.figure(figsize=(6, 3))
# plt.plot(
#     np.arange(c.t_steps) * c.dt, m2.d.u_max, label="FTCS Oscillation Amplitude"
# )
# plt.yscale("log")
# start = int(c.t_steps / 2)
# lr = stats.linregress(c.time_axis[start:], np.log(m2.d.u_max[start:]))
# plt.plot(
#     c.time_axis,
#     np.exp(lr.slope * c.time_axis + lr.intercept),
#     "r--",
#     linewidth=0.5,
#     label="regression: $A = exp(1.41t - 7.24)$",
# )
# plt.xlim(c.time_axis[0], c.time_axis[-1])
# plt.xlabel(r"$t$")
# plt.ylabel(r"$A^n$")
# plt.legend()
# plt.tight_layout()
# print(f"G: {np.exp([lr.slope])}. Expected: {math.sqrt(1 + (c.c*c.dt/c.dx)**2)}")
# save_plot("ftcs-error-growth.pdf")

c = UpwindConfiguration(M=10001, dt=0.09, c=10)
m3 = AdvectionModel(c)
m3.run()
plots.plot_snapshots(
    m3,
    plot_title="Simple Upwind Method",
    snapshot_times=[0.5 * c.t_max],
    hold=False,
)
a3 = plots.animate_fd(
    m3, plot_title="Simple Upwind Method", repeat=True, hold=False
)

c = LaxConfiguration(M=10001, dt=0.09, c=10)
m4 = AdvectionModel(c)
m4.run()
plots.plot_snapshots(
    m4, plot_title="Lax Method", snapshot_times=[0.5 * c.t_max], hold=False
)
# a4 = plots.animate_fd(m4, plot_title="Lax Method", repeat=True, hold=False)

c = LaxWendroffConfiguration(M=10001, dt=0.09, c=10)
m5 = AdvectionModel(c)
m5.run()
plots.plot_snapshots(
    m5,
    plot_title="Lax-Wendroff Method",
    snapshot_times=[0.5 * c.t_max],
    hold=False,
)
# a5 = plots.animate_fd(
#     m5, plot_title="Lax-Wendroff Method", repeat=True, hold=False
# )

plt.show()
