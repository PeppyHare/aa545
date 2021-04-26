#!/usr/bin/env python3
"""
Coding project 1: Electrostatic Particle in Cell.
"""
import math
import os
from pathlib import Path
import datetime
import random

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
import numpy as np
import progressbar
from scipy import stats

from weighting import weight_particles, weight_field
from poisson import setup_poisson, compute_field


###############################################################################
# Utility functions
###############################################################################
def create_folder(path):
    """Create a (possibly nested) folder if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    create_folder(os.path.join(os.getcwd(), "plots", "pic1"))
    fig_name = os.path.join("plots", "pic1", filename)
    plt.savefig(fig_name)
    print(f"Saved figure {os.path.join(os.getcwd(), fig_name)} to disk.")


def count_crossings(arr):
    """Count the number of times the value in numpy.array :arr: crosses its
    average value"""
    avg = np.average(arr)
    is_above = arr[0] > avg
    crossings = 0
    for n in range(1, arr.shape[0]):
        if is_above and arr[n] < avg:
            is_above = False
            crossings += 1
        elif (not is_above) and arr[n] > avg:
            is_above = True
            crossings += 1
    return crossings


class Configuration:
    """Default configuration for PIC model.

    Configuration parameters in this struct do not change over the course of
    the simulation.
    """

    # Total number of particles
    N = 512
    # Number of grid cells (number of grid points - 1)
    M = 128
    # Left-hand boundary of periodic domain
    x_min = -np.pi
    # Right-hand boundary of periodic domain
    x_max = np.pi
    # Min bound for plotting velocity
    v_min = -2
    # Max bound for plotting velocity
    v_max = 2
    # Normalization of electric constant
    eps0 = 1.0
    # Charge-to-mass ratio of species
    qm = -1
    # Plasma frequency
    wp = 1
    # Time step
    dt = 0.01
    # Number of periods of the plasma frequency to integrate in time
    n_periods = 20
    # Maximum value of time we will integrate to
    t_max = n_periods * (2 * np.pi / wp)
    # Weighting order.  0 = nearest grid point. 1 = linear weighting
    weighting_order = 1
    # Should the animation repeat?
    repeat_animation = False
    # Particle plotting size. 1 is small, 20 is large.
    markersize = 2
    # Whether to plot grid lines
    plot_grid_lines = False
    # If plotting snapshots, do so at these values of t
    snapshot_times = [0, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max]
    # If false, disable live plots of energy
    plot_energy = False
    # If false, disable live plots of electric field
    plot_fields = False

    def __init__(self):
        """Return a valid configuration for PICModel."""
        # Particle charge
        self.q = (self.wp ** 2) * self.eps0 / (self.N * self.qm)
        # Background charge density
        self.rho_bg = -self.N * self.q
        # Set the initial position/velocity of all particles.
        self.initialize_particles()
        # Set phase space ranges used in plots
        self.x_range = (self.x_min, self.x_max)
        self.v_range = (self.v_min, self.v_max)
        # Scale position from [x_min, x_max] to [0, 1]
        # x' = (x - x_min)/(x_max - x_min)
        self.L = self.x_max - self.x_min
        # Total number of time steps
        self.t_steps = math.ceil(self.t_max / self.dt)
        # Time axis used for plots
        self.time_axis = np.linspace(0, self.t_max, self.t_steps)
        # Grid spacing / particle size
        self.dx = 1 / self.M
        # Particle mass
        self.m = self.q / self.qm
        # Number of subsampled frames to hold in memory
        self.history_steps = min(self.t_steps, 5000)
        # State of all particles is stored every {subsample_ratio} time steps
        self.subsample_ratio = math.ceil(self.t_steps / self.history_steps)
        # Finite difference matrix used to solve Poisson equation
        (self.inv_a, _) = setup_poisson(self.M)

    def initialize_particles(self):
        """Set the initial position of all particles in phase space.

        Positions/velocities here are absolute, and not scaled to [0,1].
        """
        # Default: Maxwellian beam
        v_fwhm = 2
        v_stdev = v_fwhm / 2.355
        random.seed("not really random")
        initial_x = np.zeros(self.N)
        initial_v = np.zeros(self.N)
        for i in range(self.N):
            initial_x[i] = random.uniform(self.x_min, self.x_max)
            initial_v[i] = max(
                min(random.gauss(0, v_stdev), self.v_max), self.v_min
            )
        self.initial_x = initial_x
        self.initial_v = initial_v


class ParticleData:
    """Struct containing the state and history of simulation data.

    The data contained in these objects & vectors will change over the course
    of the simulation.
    """

    def __init__(self, c: Configuration):
        # Grid points
        self.x_j = np.linspace(0, 1, c.M + 1)[:-1]
        # Calculate total kinetic energy at each time step
        self.ke_hist = np.zeros(c.t_steps)
        # Calculate total electric field energy at each time step
        self.fe_hist = np.zeros(c.t_steps)
        # Calculate total momentum at each time step
        self.p_hist = np.zeros(c.t_steps)
        # Current position of each particle
        self.x_i = np.zeros(c.N)
        # Current velocity of each particle
        self.v_i = np.zeros(c.N)
        # Current value of electric field at grid points
        self.e_j = np.zeros(c.M)
        # Maximum value of total energy seen so far
        self.energy_max = 0
        # Maximum value of electric field seen so far
        self.efield_max = 0
        # Subsampled history of particle positions
        self.x_hist = np.zeros((c.N, c.history_steps + 3))
        # Subsampled history of particle velocities
        self.v_hist = np.zeros((c.N, c.history_steps + 3))


class PICModel:
    def __init__(self, c: Configuration):
        self.c = c
        d = ParticleData(c)

        # Normalize particle positions to [0, 1]
        d.x_i = (c.initial_x - c.x_min) / c.L
        d.v_i = c.initial_v / c.L

        # Initialize leapfrog with a 1/2 step backwards
        rho = (
            c.eps0
            * weight_particles(
                d.x_i, d.x_j, c.dx, c.M, c.q, order=c.weighting_order
            )
            + c.rho_bg
        )
        # Solve for field at t=0
        d.e_j = compute_field(rho, c.inv_a, c.dx)
        # Weight field to grid and accelerate particles
        e_i = weight_field(d.x_i, d.x_j, d.e_j, c.dx, order=c.weighting_order)
        d.v_i -= (c.dt / 2) * c.qm * e_i

        self.d = d

    @staticmethod
    @numba.njit
    def time_step(
        frame,
        x_i,
        v_i,
        x_j,
        ke_hist,
        fe_hist,
        p_hist,
        x_hist,
        v_hist,
        subsample_ratio,
        M,
        dx,
        dt,
        eps0,
        rho_bg,
        q,
        qm,
        m,
        weighting_order,
        inv_a,
        nonumba=False,
    ):
        """Evolve state forward in time by ∆t with periodic boundary conditions.

        Governing equations are:
            x[i](t + ∆t) = x[i](t) + v[i](t + ∆t/2) * ∆t
            v[i](t + ∆t) = v[i](t) + e[i](t + ∆t/2) * (q/m) * ∆t

        The @numba.jit decorator compiles the particle_push function to optimized
        machine code using the `llvmlite` version of the LLVM compiler. Depending on
        the size of n and t_steps, the performance improvement is up to 100x the
        speed of the pure Python (still awful compared to C, but good enough for
        educational purposes).

        The steps are:
        1. Weight particle positions to the grid.
        2. Solve for fields at the grid points. Compute field energy.
        3. Weight fields to particles.
        4. Half-accelerate velocity. Compute kinetic energy, momentum.
        5. Half-accelerate velocity.
        6. Push position.
        """
        rho = (
            eps0 * weight_particles(x_i, x_j, dx, M, q, order=weighting_order)
            + rho_bg
        )

        # Solve Poisson's equation
        e_j = compute_field(rho, inv_a, dx)

        # Calculate total electric field energy
        fe_hist[frame] += dx / 2 * eps0 * np.sum(e_j * e_j)

        # Weight field on grid to particles
        e_i = weight_field(x_i, x_j, e_j, dx, order=weighting_order)

        # Calculate what acceleration will be
        dv = dt * qm * e_i

        # Compute kinetic energy, momentum
        ke_hist[frame] += m / 2 * np.sum(np.abs(v_i * (v_i + dv)))
        p_hist[frame] += m * np.sum(v_i + (dv / 2))

        # Accelerate and push
        # x[i] and v[i] are offset in time by ∆t/2, so that they leap-frog past each
        # other:
        #
        #          x(old)       x(new)
        # -----------*------------*----->
        #   v(old)       v(new)
        # ----*------------*------------> t
        #     |      |     |      |
        #   -∆t/2    0   ∆t/2    ∆t

        # Accelerate
        v_i += dv

        # Push particles forward, now that we have v_i[n+1/2]
        # print(frame, subsample_ratio, int(frame/subsample_ratio) - 1, ke_hist.size, x_hist.shape)
        for i in numba.prange(x_i.size):
            # numba.prange is like np.arange, but optimized for parallelization
            # across multiple CPU cores

            x_i[i] += v_i[i] * dt
            # Apply periodic boundary conditions. NumPy uses the definition of floor
            # where floor(-2.5) == -3.
            x_i[i] -= np.floor(x_i[i])
            if frame % subsample_ratio == 0:
                x_hist[i][int(frame / subsample_ratio)] += x_i[i]
                v_hist[i][int(frame / subsample_ratio)] += v_i[i]

    def run(self, showprogress=True, debugprint=False):
        """Run the simulation."""
        c = self.c
        d = self.d
        if showprogress:
            print("Simulating...")
            bar = progressbar.ProgressBar(
                maxval=c.t_steps,
                widgets=[
                    progressbar.Bar("=", "[", "]"),
                    " ",
                    progressbar.Percentage(),
                ],
            )
            bar.start()
        if debugprint:
            print("Running with the following params:")
            print(f"t_max: {c.t_max}")
            print(f"t_steps: {c.t_steps}")
            print(f"dt: {c.dt}")
            print(f"N: {c.N}")
            print(f"M: {c.M}")
            print(f"dx: {c.dx}")
            print(f"wp: {c.wp}")
            print(f"q: {c.q}")
            print(f"m: {c.m}")
            print(f"rho_bg: {c.rho_bg}")
        for frame in range(c.t_steps):
            self.time_step(
                frame,
                x_i=d.x_i,
                v_i=d.v_i,
                x_j=d.x_j,
                ke_hist=d.ke_hist,
                fe_hist=d.fe_hist,
                p_hist=d.p_hist,
                x_hist=d.x_hist,
                v_hist=d.v_hist,
                subsample_ratio=c.subsample_ratio,
                M=c.M,
                dx=c.dx,
                dt=c.dt,
                eps0=c.eps0,
                rho_bg=c.rho_bg,
                q=c.q,
                qm=c.qm,
                m=c.m,
                weighting_order=c.weighting_order,
                inv_a=c.inv_a,
            )
            if showprogress:
                bar.update(frame)
        if showprogress:
            bar.finish()
            print("done!")


# ###############################################################################
# # Animation code
# ###############################################################################
# def init_animation():
#     """Animation initialization."""
#     global ax_xv, pt_xv1, pt_xv2, x_range, v_min, v_max
#     ax_xv.set_xlim(x_range)
#     ax_xv.set_ylim(v_min, v_max)
#     return (pt_xv1, pt_xv2)


# def animate(frame):
#     """Call every time we update animation frame."""
#     global x_i, v_i, dt, ke_hist, L, x_min, x_max, t_steps
#     global pt_ke, pt_fe, pt_te, time_axis, pt_xv1, pt_xv2, time_text, ax_energy
#     global ke_hist, fe_hist, p_hist, x_hist, v_hist
#     global x_j, e_j, pt_efield, energy_max, efield_max, plot_energy, plot_fields
#     if frame == 0:
#         x_i, v_i, ke_hist = initialize()
#     x_i, v_i, e_j = time_step(
#         frame,
#         x_i=x_i,
#         v_i=v_i,
#         x_j=x_j,
#         ke_hist=ke_hist,
#         fe_hist=fe_hist,
#         p_hist=p_hist,
#         x_hist=x_hist,
#         v_hist=v_hist,
#         subsample_ratio=subsample_ratio,
#         nonumba=False,
#     )
#     if frame % subsample_ratio == 0:
#         current_time = frame * dt
#         time_text.set_text(f"t = {current_time:.2f}")
#         n2 = math.ceil(N / 2)
#         pt_xv1.set_data((x_i[:n2] * L) + x_min, (v_i[:n2] * L))
#         if n2 > 0:
#             pt_xv2.set_data((x_i[n2:] * L) + x_min, (v_i[n2:] * L))
#         if plot_fields:
#             if np.max(np.abs(e_j)) > 2 * efield_max:
#                 efield_max += np.max(np.abs(e_j))
#                 ax_efield.set_ylim(-efield_max * 2, efield_max * 2)
#             pt_efield.set_data(
#                 (np.concatenate([x_j, np.array([1.0])]) * L) + x_min,
#                 np.concatenate([e_j, e_j[0:1]]),
#             )
#         if plot_energy:
#             if fe_hist[frame] + ke_hist[frame] > 2 * energy_max:
#                 energy_max += fe_hist[frame] + ke_hist[frame]
#                 ax_energy.set_ylim(0, energy_max * 2)
#             pt_ke.set_data(time_axis, ke_hist)
#             pt_fe.set_data(time_axis, fe_hist)
#             pt_te.set_data(time_axis, fe_hist + ke_hist)
#     return pt_xv1, pt_xv2, time_text, pt_ke, pt_fe, pt_te, pt_efield


# ###############################################################################
# # Main script
# ###############################################################################
# if "plot_initial_distributions" in step_flags:
#     print("Generating plots of initial particle state.")
#     x_i, v_i, ke_hist = initialize()
#     bin_width = 0.1

#     fig1 = plt.figure(figsize=(12, 8))
#     fig1.suptitle(f"Initial distribution (n={N})")

#     # Plot initial position histogram
#     rho_weight_ngp = weight_particles(x_i, x_j, dx, M, order=0)
#     print(f"Total charge (ngp): {np.sum(rho_weight_ngp[:-1] * dx)}")
#     rho_weight_lin = weight_particles(x_i, x_j, dx, M, order=1)
#     print(f"Total charge (linear): {np.sum(rho_weight_lin[:-1] * dx)}")
#     ax_init_position = fig1.add_subplot(2, 2, 1)
#     bins = math.ceil((x_range[1] - x_range[0]) / bin_width)
#     # ax_init_position.hist((x_i * L) + x_min, bins=bins, range=x_range)
#     (pt_ngp,) = ax_init_position.step(
#         (x_j * L) + x_min,
#         rho_weight_ngp,
#         color="r",
#         marker="o",
#         where="mid",
#         linewidth=2,
#         label="NGP",
#         markersize=5,
#     )
#     (pt_linear,) = ax_init_position.plot(
#         (x_j * L) + x_min,
#         rho_weight_lin,
#         color="g",
#         marker="o",
#         linestyle="--",
#         linewidth=2,
#         label="Linear",
#         markersize=5,
#     )
#     ax_init_position.legend(
#         [pt_ngp, pt_linear],
#         [pt_ngp.get_label(), pt_linear.get_label()],
#     )
#     ax_init_position.set_ylim(bottom=0)
#     ax_init_position.set_xlabel(r"$x$")
#     ax_init_position.set_ylabel("Density")
#     plt.xlim(x_range)
#     plt.title("Position")

#     # Plot initial velocity histogram
#     ax_init_velocity = fig1.add_subplot(2, 2, 2)
#     bins = math.ceil((v_range[1] - v_range[0]) / bin_width)
#     ax_init_velocity.hist(v_i * L, bins=bins, range=v_range)
#     ax_init_velocity.set_xlabel(r"$v$")
#     plt.xlim(x_range)
#     plt.title("Velocity")

#     # Plot initial positions in phase space
#     ax_init_phase = fig1.add_subplot(2, 2, (3, 4))
#     plt.title("Initial phase space")
#     n2 = math.ceil(N / 2)
#     ax_init_phase.plot(
#         (x_i[:n2] * L) + x_min,
#         (v_i[:n2] * L),
#         ".",
#         color="tab:orange",
#         markersize=markersize,
#         label="xv",
#     )
#     ax_init_phase.plot(
#         (x_i[n2:] * L) + x_min,
#         (v_i[n2:] * L),
#         ".",
#         color="tab:cyan",
#         markersize=markersize,
#         label="xv",
#     )
#     plt.xlim(x_range)
#     ax_init_phase.set_xlabel(r"$x$")
#     ax_init_phase.set_ylabel(r"$v$")
#     # Plot grid points
#     if plot_grid_lines:
#         for grid_pt in x_j:
#             ax_init_phase.axvline(
#                 (grid_pt * L) + x_min,
#                 linestyle="--",
#                 color="k",
#                 linewidth=0.2,
#             )
#             ax_init_position.axvline(
#                 (grid_pt * L) + x_min,
#                 linestyle="--",
#                 color="k",
#                 linewidth=0.2,
#             )
#     plt.tight_layout()
#     save_plot(f"initial_hist_{N}_particles.pdf")
#     plt.show()  # Waits for user to close the plot

# if "animate_phase_space" in step_flags:
#     print("Generating animation of phase space over time.")
#     initialize()
#     bigfig = plt.figure(figsize=(12, 8))
#     bigfig.suptitle(f"n={N}  m={M}  dt={dt:.4f} t_max={t_max:.4f}")
#     ax_xv = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
#     ax_energy = plt.subplot2grid((2, 2), (1, 0))
#     ax_efield = plt.subplot2grid((2, 2), (1, 1))

#     ax_xv.set_title("Phase space animation")
#     ax_xv.set_ylabel("v")
#     ax_xv.set_xlabel("x")
#     ax_xv.set_xlim(x_range)
#     ax_xv.set_ylim(v_range)

#     ax_energy.set_title("Energy")
#     ax_energy.set_xlim(0, t_max)
#     # ax_energy.set_ylim(0, 2 * (fe_hist[0] + ke_hist[0]))
#     ax_energy.set_ylabel("energy")
#     ax_energy.set_xlabel("time")

#     ax_efield.set_title("Fields")
#     ax_efield.set_xlim(x_range)
#     ax_efield.set_ylim(v_range)
#     ax_efield.set_ylabel("Electric field")
#     ax_efield.set_xlabel("x")

#     (pt_xv1,) = ax_xv.plot(
#         [],
#         [],
#         ".",
#         color="tab:orange",
#         markersize=markersize,
#         label="xv",
#     )
#     (pt_xv2,) = ax_xv.plot(
#         [],
#         [],
#         ".",
#         color="tab:cyan",
#         markersize=markersize,
#         label="xv",
#     )
#     (pt_ke,) = ax_energy.plot([], [], "b-", markersize=1, label="ke")
#     (pt_fe,) = ax_energy.plot([], [], "g-", markersize=1, label="fe")
#     (pt_te,) = ax_energy.plot([], [], "k-", markersize=1, label="total")
#     ax_energy.legend(
#         [pt_ke, pt_fe, pt_te],
#         [pt_ke.get_label(), pt_fe.get_label(), pt_te.get_label()],
#     )
#     (pt_efield,) = ax_efield.plot([], [], "c", label="efield")

#     # Add the grid points to the plot of the fields, but only if there aren't
#     # too many of them
#     if x_j.size < 80 and plot_grid_lines:
#         for grid_pt in x_j:
#             ax_efield.axvline(
#                 (grid_pt * L) + x_min,
#                 linestyle="--",
#                 color="k",
#                 linewidth=0.2,
#             )

#     # Add a label to the frame showing the current time. Updated each time step
#     # in update()
#     time_text = ax_xv.text(0.02, 0.95, "", transform=ax_xv.transAxes)

#     # Evolve positions until t_max. Animate particle positions in phase space.
#     animation = FuncAnimation(
#         bigfig,
#         animate,
#         frames=t_steps,
#         init_func=init_animation,
#         blit=True,
#         interval=1,
#         repeat=repeat_animation,
#     )
#     plt.tight_layout()
#     plt.show()  # Waits for user to close the plot

# if "plot_snapshots" in step_flags:
#     print("Generating snapshots of state at various time intervals.")
#     t_steps = math.ceil(t_max / dt)
#     initialize()
#     snapshot_times.sort()
#     snapshot_frames = [math.floor(t / dt) for t in snapshot_times]
#     snapshots = []
#     print("Simulating...")
#     bar = progressbar.ProgressBar(
#         maxval=t_steps,
#         widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
#     )
#     bar.start()
#     for frame in range(t_steps):
#         rho = weight_particles(x_i, x_j, dx, M, order=weighting_order) + rho_bg
#         e_j = compute_field(rho, inv_a, dx)
#         e_i = weight_field(x_i, x_j, e_j, dx, order=weighting_order)
#         time_step(
#             frame,
#             x_i=x_i,
#             v_i=v_i,
#             x_j=x_j,
#             ke_hist=ke_hist,
#             fe_hist=fe_hist,
#             p_hist=p_hist,
#             x_hist=x_hist,
#             v_hist=v_hist,
#             subsample_ratio=subsample_ratio,
#             nonumba=False,
#         )
#         if frame in snapshot_frames:
#             snapshots.append(
#                 {
#                     "x": x_hist[:, int(frame / subsample_ratio)],
#                     "v": v_hist[:, int(frame / subsample_ratio)],
#                     "frame": frame,
#                 }
#             )
#         bar.update(frame + 1)
#     bar.finish()
#     print(f"Sampled {len(snapshots)} snapshots over {t_steps} time steps.")
#     fig3 = plt.figure()
#     fig3.suptitle(f"Time snapshots (n={N})")
#     num_subplots = len(snapshots)
#     idx = 1
#     for snapshot in snapshots:
#         ax_xv = fig3.add_subplot(num_subplots, 1, idx)
#         cur_t = snapshot["frame"] * dt
#         ax_xv.set_ylabel("v")
#         ax_xv.set_title(f"t={cur_t:.2f}")
#         plt.plot(
#             (snapshot["x"] * L) + x_min,
#             (snapshot["v"] * L),
#             "ko",
#             markersize=1,
#         )
#         plt.xlim(x_range)
#         if idx == num_subplots:
#             ax_xv.set_xlabel("x")
#         idx += 1
#     plt.tight_layout()
#     save_plot(f"snapshots_{N}_particles.pdf")

#     plt.show()  # Waits for user to close the plots

# if "no_plots" in step_flags:
#     initialize()
#     run(showprogress=True)
#     print(f"crossings(ke): {count_crossings(ke_hist)}")
#     print(f"crossings(x[0]): {count_crossings(x_hist[0,:])}")
#     print(f"t_max/2π: {t_max/(2*np.pi)}")
#     print(f"dt*wp/2: {dt*wp/2:.4f}")
#     print(
#         "measured w*dt/2:"
#         f" {dt*count_crossings(ke_hist)/(4*t_max/(2*np.pi)):.4f}"
#     )

# if "plot_dispersion" in step_flags:
#     n_trials = 10
#     w_measured = np.zeros(n_trials)
#     dt_trials = np.linspace(1, 10, n_trials) * 0.5 * (2 / wp)
#     bar = progressbar.ProgressBar(
#         maxval=n_trials,
#         widgets=[
#             progressbar.Bar("=", "[", "]"),
#             " ",
#             progressbar.Percentage(),
#         ],
#     )
#     bar.start()
#     for idx in range(n_trials):
#         dt = dt_trials[idx]
#         t_steps = math.ceil(t_max / dt)
#         initialize()
#         run(showprogress=False)
#         w_measured[idx] = count_crossings(ke_hist) / (4 * t_max / (2 * np.pi))
#         # print("Crossing counts:")
#         # print(count_crossings(ke_hist))
#         # print(count_crossings(fe_hist))
#         # print(count_crossings(x_hist[0, :]))
#         bar.update(idx)
#     bar.finish()
#     plt.figure()
#     plt.plot((dt_trials * wp / 2), (w_measured * dt / 2), "k.")
#     plt.ylabel("w*dt/2")
#     plt.xlabel("wp*dt/2")
#     plt.show()


# if "trace_particles" in step_flags:
#     print("Generating trace plots of particles in phase space.")
#     initialize()
#     run()
#     fig4 = plt.figure(figsize=(12, 8))
#     fig4.suptitle(
#         f"Particle Traces (n={N}, wp*dt={wp*dt:.3f}, 1st-order weighting)"
#     )
#     ax4 = fig4.add_subplot(211)
#     for i in range(N):
#         position = (x_hist[i] * L) + x_min
#         velocity = v_hist[i] * L
#         ax4.plot(position, velocity, ".", markersize=3)
#         ax4.set_xlabel("x")
#         ax4.set_ylabel("v")
#         ax4.set_title("Particle Traces")
#     ax_energy = fig4.add_subplot(212)
#     ax_energy.set_title("Energy")
#     ax_energy.set_xlim(0, t_max)
#     # ax_energy.set_ylim(0, 2 * (fe_hist[0] + ke_hist[0]))
#     ax_energy.set_ylabel("energy")
#     ax_energy.set_xlabel("time")
#     (pt_ke,) = ax_energy.plot(
#         time_axis, ke_hist, "b-", markersize=1, label="ke"
#     )
#     (pt_fe,) = ax_energy.plot(
#         time_axis, fe_hist, "g-", markersize=1, label="fe"
#     )
#     (pt_te,) = ax_energy.plot(
#         time_axis, ke_hist + fe_hist, "k-", markersize=1, label="total"
#     )
#     ax_energy.set_ylim(0, 1.2 * np.max(ke_hist + fe_hist))
#     ax_energy.legend(
#         [pt_ke, pt_fe, pt_te],
#         [pt_ke.get_label(), pt_fe.get_label(), pt_te.get_label()],
#         loc="upper right",
#     )
#     plt.tight_layout()
#     save_plot(f"traces_{N}_particles.pdf")

#     plt.show()  # Waits for user to close the plots


# if "plot_energy" in step_flags:
#     print("Generating only energy plots.")
#     initialize()
#     run()
#     fig4 = plt.figure(figsize=(12, 6))
#     fig4.suptitle(
#         f"Time Evolution (n={N}, wp*dt={wp*dt:.3f}, 1st-order weighting)"
#     )
#     ax_energy = fig4.add_subplot(2, 1, (1, 2))
#     ax_energy.set_title("Energy")
#     ax_energy.set_xlim(0, t_max)
#     ax_energy.set_ylabel("energy")
#     ax_energy.set_xlabel("time")
#     (pt_te,) = ax_energy.plot(
#         time_axis, np.log(ke_hist + fe_hist), "k-", markersize=1, label="total"
#     )
#     (pt_ke,) = ax_energy.plot(
#         time_axis, np.log(ke_hist), "b-", markersize=1, label="ke"
#     )
#     (pt_fe,) = ax_energy.plot(
#         time_axis, np.log(fe_hist), "g-", markersize=1, label="fe"
#     )
#     # ax_energy.set_ylim(0, 1.2 * max(np.log(ke_hist + fe_hist)))
#     ax_energy.legend(
#         [pt_ke, pt_fe, pt_te],
#         [pt_ke.get_label(), pt_fe.get_label(), pt_te.get_label()],
#         loc="upper right",
#     )
#     plt.tight_layout()
#     save_plot(f"energy_wpdt_{wp*dt/2:.3f}.pdf")

#     # Plot of field energy only
#     plt.figure()
#     ax = plt.gca()
#     ax.set_title("Field energy")
#     ax.set_xlim(0, t_max)
#     ax.set_ylabel("ln(E_field)")
#     ax.set_xlabel("time")
#     y = np.log(np.sqrt(fe_hist))
#     ax.plot(time_axis, y, "g-", markersize=1)

#     # Plot of ratio between field energy to kinetic energy
#     plt.figure()
#     ax = plt.gca()
#     ax.set_title("Field energy")
#     ax.set_xlim(0, t_max)
#     ax.set_ylabel("E_field/E_kinetic")
#     ax.set_xlabel("time")
#     y = fe_hist / ke_hist
#     ax.plot(time_axis, y, "g-", markersize=1)

#     plt.show()  # Waits for user to close the plots

# if "coupled_oscillators" in step_flags:
#     v0_trials = np.linspace(0, 4, 40)[1:]
#     results = np.zeros_like(v0_trials)
#     for idx in range(v0_trials.size):
#         initial_v[0] = v0_trials[idx]
#         initial_v[1] = -v0_trials[idx]
#         initialize()
#         run(showprogress=True)
#         print(f"crossings(ke): {count_crossings(ke_hist)}")
#         print(f"crossings(x[0]): {count_crossings(x_hist[0,:])}")
#         print(f"t_max/2π: {t_max/(2*np.pi)}")
#         print(f"dt*wp: {dt*wp:.2f}")
#         print(f"measured wp: {count_crossings(ke_hist)/(4*t_max/(2*np.pi))}")
#         results[idx] = count_crossings(ke_hist) / (4 * t_max / (2 * np.pi))
#     expected = v0_trials
#     fig = plt.figure()
#     fig.suptitle("SHM with initial velocity")
#     ax = fig.add_subplot(111)
#     (pt_res,) = ax.plot(v0_trials, results, "k.", markersize=5, label="results")
#     (pt_exp,) = ax.plot(v0_trials, expected, "r--", label="expected")
#     ax.set_xlabel("v0")
#     ax.set_ylabel("Measured frequency")
#     plt.legend([pt_res, pt_exp], ["Measured frequency", "v0"])
#     save_plot("coupled_oscillators_dispersion.pdf")
#     plt.show()

# if "instability_growth_regression" in step_flags:
#     print("Trying to measure instability.")
#     initialize()
#     run()
#     # Try and find where field energy goes above 25% of total energy
#     ramp_start = 0
#     ramp_end = t_steps - 1
#     for idx in range(t_steps):
#         if fe_hist[idx] / ke_hist[idx] > 0.001:
#             ramp_start = idx
#             break
#     for idx in range(t_steps):
#         if fe_hist[idx] / ke_hist[idx] > 0.1:
#             ramp_end = idx
#             break
#     print(
#         f"Instability occurred between steps {ramp_start} and {ramp_end} out of"
#         f" {t_steps}."
#     )
#     lr = stats.linregress(
#         time_axis[ramp_start:ramp_end], np.log(fe_hist[ramp_start:ramp_end])
#     )
#     print(lr)
#     fig4 = plt.figure(figsize=(12, 6))
#     fig4.suptitle(
#         f"Time Evolution (n={N}, wp*dt={wp*dt:.3f}, 1st-order weighting)"
#     )
#     ax_energy = fig4.add_subplot(2, 1, (1, 2))
#     ax_energy.set_title("Energy")
#     ax_energy.set_xlim(0, t_max)
#     ax_energy.set_ylabel("energy")
#     ax_energy.set_xlabel("time")
#     (pt_te,) = ax_energy.plot(
#         time_axis, ke_hist + fe_hist, "k-", markersize=1, label="total"
#     )
#     (pt_ke,) = ax_energy.plot(
#         time_axis, ke_hist, "b-", markersize=1, label="ke"
#     )
#     (pt_fe,) = ax_energy.plot(
#         time_axis, fe_hist, "g-", markersize=1, label="fe"
#     )
#     plt.yscale("log")
#     # ax_energy.set_ylim(0, 1.2 * max(np.log(ke_hist + fe_hist)))
#     ax_energy.legend(
#         [pt_ke, pt_fe, pt_te],
#         [pt_ke.get_label(), pt_fe.get_label(), pt_te.get_label()],
#         loc="upper right",
#     )
#     plt.tight_layout()
#     save_plot(
#         f"energy_instability_growth_{datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}.pdf"
#     )

#     # plt.show()  # Waits for user to close the plots
