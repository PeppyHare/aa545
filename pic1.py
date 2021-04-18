#!/usr/bin/env python3
"""
Coding project 1.1: Kinetic Modeling.

Write a code that evolves free streaming particles in phase space, (x, vx).
Assume there are no forces and no collisions. Track the position and velocity of
each particle. The code should work with an arbitrary number of particles.
Implement periodic boundary conditions in the x direction. Visualize the motion
of the particles by plotting their positions in phase space as a function of
time. Also, plot particle trajectories by not erasing their positions from the
previous time. Implement a diagnostic that computes the time history of the
kinetic energy of all particles.

Initialize a randomly filled particle distribution that is Maxwellian in vx and
is uniform in x using N = 128, 512, 2048. The FWHM in velocity should be 2. Use
a domain that is vx = [-5, 5] and x = [-2π, 2π]. For the three cases, generate a
histogram of the particle density as a function of vx, using a bin width of
0.25. Evolve the particle motion until t = 8π. Produce plots of the particle
positions for the N = 512 case at t = 0, 2π, & 8π. Show the plot of the particle
trajectories for the N = 128 case up to t = 2π. Compare plots of the kinetic
energy history for all four cases for different time step values. Upload your
report in pdf format, source code, and run script/instructions.
"""
import math
import os
from pathlib import Path
import random
import time

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
import numpy as np
import progressbar

from weighting import weight_particles, weight_field
from poisson import setup_poisson, solve_poisson


###############################################################################
# Settings
###############################################################################

# "plot_initial_distributions": Plot histograms of the initial position and
# velocity and plot the initial state in phase space

# "animate_phase_space": Create an animation of phase space over time Mutually
# exclusive with generating timeseries plots

# "plot_snapshots": Plot snapshots of phase space at various times

# "trace_particles": Plot traces of particles in phase space over time

# "compare_ke": Run simulation using various values for the time step and plot
# the change in the total kinetic energy diagnostic over time

# "performance_testing": Run the simulation with many particles and a small time
# step to evaluate the elapsed per-step computation time
step_flags = [
    # "plot_initial_distributions",
    "animate_phase_space",
    # "plot_snapshots",
    # "trace_particles",
    # "compare_ke",
    # "performance_testing",
]


# Number of particles
n = 128
# n = 512
# n = 2048

# Number of grid points
m = 33

# Initial position distribution: uniform over [-2π, 2π]
x_min = -2 * np.pi
x_max = 2 * np.pi
# Initial velocity distribution: Maxwellian with FWHM=2
v_fwhm = 2
v_min = -5
v_max = 5

# Time step and duration
dt = 0.05
t_max = 80 * np.pi

# Weighting order. 0 = nearest grid point. 1 = linear weighting
weighting_order = 1

# Charge-to-mass ratio of species
qm = 0.1

###############################################################################
# Initialization
###############################################################################
x_range = (x_min, x_max)
v_range = (v_min, v_max)

# Scale position from [x_min, x_max] to [0, 1]
# x' = (x - x_min)/(x_max - x_min)
x_scale = x_max - x_min

# For normal distribution, FWHM = 2*Sqrt(2*ln(2))σ ~ 2.355σ
v_stdev = v_fwhm / 2.355

t_steps = math.ceil(t_max / dt)
frame = 0  # the current time step

# Grid spacing / particle size
dx = 1 / (m - 1)

# Initialize arrays
x = np.zeros(n)
v = np.zeros(n)
e_j = np.zeros(m)

# Store the history of all particles
x_history = np.zeros((n, t_steps))
v_history = np.zeros((n, t_steps))

grid_pts = np.linspace(0, 1, m)

# Calculate the kinetic energy at each timestep
ke = np.zeros(t_steps)

# Finite difference matrix used to solve Poisson equation
inv_a = setup_poisson(m)

# Background charge density
rho_bg = -n / ((m - 1) * dx)


def initialize_state():
    """Set initial positions of all particles."""
    global x, v, ke, x_scale, v_min, v_max, v_stdev, t_steps, x_history, v_history
    # Fixing random state for reproducibility
    random.seed("not really random")
    initial_x = np.zeros(n)
    initial_v = np.zeros(n)
    for i in range(n):
        initial_x[i] = random.uniform(0, 1)
        initial_v[i] = (
            max(min(random.gauss(0, v_stdev), v_max), v_min) / x_scale
        )
    x = initial_x
    v = initial_v

    # Project velocity backwards 1/2 time step
    rho = weight_particles(x, grid_pts, dx, m, order=weighting_order) + rho_bg
    # Solve for field at t=0
    e_j = solve_poisson(rho, inv_a, dx)
    e_i = weight_field(x, grid_pts, e_j, dx, order=weighting_order)
    v += (-dt / 2) * qm / x_scale * e_i

    # (Re-)initialize state history
    x_history = np.zeros((n, t_steps))
    v_history = np.zeros((n, t_steps))

    # (Re-)initialize total kinetic energy
    ke = np.zeros(t_steps)

    # Set the number of time steps
    t_steps = math.ceil(t_max / dt)

    return x, v, ke


###############################################################################
# Time step
###############################################################################
@numba.njit
def particle_push(x, v, frame, dt, e_i, ke, x_history, v_history):
    """Evolve state forward in time by ∆t with periodic boundary conditions.

    Governing equations are: x[i](t + ∆t) = x[i](t) + v[i](t + ∆t/2) * ∆t v[i](t
        + ∆t) = v[i](t) + e[i](t + ∆t/2) * (q/m) * ∆t

    The @numba.jit decorator compiles the particle_push function to optimized
    machine code using the `llvmlite` version of the LLVM compiler. Depending on
    the size of n and t_steps, the performance improvement is up to 100x the
    speed of the pure Python (still awful compared to C, but good enough for
    educational purposes).
    """
    # numba.prange is like np.arange, but optimized for parallelization across
    # multiple CPU cores
    for i in numba.prange(x.size):
        # x[i] and v[i] are offset in time by ∆t/2, so that they leap-frog past
        # each other:
        #
        #          x(old)       x(new)
        # -----------*------------*----->
        #   v(old)       v(new)
        # ----*------------*------------> t
        #     |      |     |      |
        #   -∆t/2    0   ∆t/2    ∆t

        x[i] += v[i] * dt
        v[i] += e_i[i] * dt * qm / x_scale
        # Apply periodic boundary conditions. NumPy uses the definition of floor
        # where floor(-2.5) == -3.
        x[i] -= np.floor(x[i])
        x_history[i][frame] += x[i]
        v_history[i][frame] += v[i]

    ke[frame] = 0.5 * (np.sum(np.square(v)))
    frame += 1
    return x, v


def run(nonumba=False):
    """Run the simulation."""
    global x, v, t_steps, dt, m, dx, grid_pts, ke, x_history, v_history
    print("Simulating...")
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for frame in range(t_steps):
        if not nonumba:
            rho = (
                weight_particles(x, grid_pts, dx, m, order=weighting_order)
                + rho_bg
            )
            e_j = solve_poisson(rho, inv_a, dx)
            e_i = weight_field(x, grid_pts, e_j, dx, order=weighting_order)
            particle_push(
                x,
                v,
                frame=frame,
                dt=dt,
                e_i=e_i,
                ke=ke,
                x_history=x_history,
                v_history=v_history,
            )
        else:
            rho = (
                weight_particles.py_func(
                    x, grid_pts, dx, m, order=weighting_order
                )
                + rho_bg
            )
            e_j = solve_poisson.py_func(rho, inv_a, dx)
            e_i = weight_field.py_func(
                x, grid_pts, e_j, dx, order=weighting_order
            )
            particle_push.py_func(
                x,
                v,
                frame=frame,
                dt=dt,
                e_i=e_i,
                ke=ke,
                x_history=x_history,
                v_history=v_history,
            )
        bar.update(frame + 1)
    bar.finish()
    print("done!")


###############################################################################
# Animation code
###############################################################################
def init_animation():
    """Animation initialization."""
    global ax, pt, x_range, v_min, v_max
    ax.set_xlim(x_range)
    ax.set_ylim(v_min, v_max)
    return (pt,)


def update(frame):
    """Call every time we update animation frame."""
    global x, v, dt, ke, time_text, pt
    if frame == 0:
        x, v, ke = initialize_state()
    rho = weight_particles(x, grid_pts, dx, m, order=weighting_order) + rho_bg
    e_j = solve_poisson(rho, inv_a, dx)
    e_i = weight_field(x, grid_pts, e_j, dx, order=weighting_order)
    particle_push(
        x,
        v,
        frame=frame,
        dt=dt,
        e_i=e_i,
        ke=ke,
        x_history=x_history,
        v_history=v_history,
    )
    current_time = frame * dt
    time_text.set_text(f"t = {current_time:.2f}")
    pt.set_data((x * x_scale) + x_min, (v * x_scale))
    return tuple([pt]) + tuple([time_text])


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


###############################################################################
# Main script
###############################################################################
if "plot_initial_distributions" in step_flags:
    print("Generating plots of initial particle state.")
    x, v, ke = initialize_state()
    bin_width = 0.1

    fig1 = plt.figure()
    fig1.suptitle(f"Initial distribution (n={n})")

    # Plot initial position histogram
    rho_weight_ngp = weight_particles(x, grid_pts, dx, m, order=0)
    print(f"Total charge (ngp): {np.sum(rho_weight_ngp[:-1] * dx)}")
    rho_weight_lin = weight_particles(x, grid_pts, dx, m, order=1)
    print(f"Total charge (linear): {np.sum(rho_weight_lin[:-1] * dx)}")
    ax_init_position = fig1.add_subplot(2, 2, 1)
    bins = math.ceil((x_range[1] - x_range[0]) / bin_width)
    ax_weighted = ax_init_position.twinx()
    ax_init_position.hist((x * x_scale) + x_min, bins=bins, range=x_range)
    ax_weighted.step(
        (grid_pts * x_scale) + x_min,
        rho_weight_ngp,
        color="r",
        marker="o",
        where="mid",
        linewidth=0.5,
    )
    ax_weighted.plot(
        (grid_pts * x_scale) + x_min,
        rho_weight_lin,
        color="g",
        marker="o",
        linestyle="--",
        linewidth=0.5,
    )
    ax_weighted.set_ylim(bottom=0)
    ax_init_position.set_xlabel(r"$x$")
    ax_init_position.set_ylabel("Count")
    plt.xlim(x_range)
    plt.title("Position")

    # Plot initial velocity histogram
    ax_init_velocity = fig1.add_subplot(2, 2, 2)
    bins = math.ceil((v_range[1] - v_range[0]) / bin_width)
    ax_init_velocity.hist(v * x_scale, bins=bins, range=v_range)
    ax_init_velocity.set_xlabel(r"$v$")
    plt.xlim(x_range)
    plt.title("Velocity")

    # Plot initial positions in phase space
    ax_init_phase = fig1.add_subplot(2, 2, (3, 4))
    plt.title("Initial phase space")
    plt.plot((x * x_scale) + x_min, (v * x_scale), "ko", markersize=1)
    plt.xlim(x_range)
    ax_init_phase.set_xlabel(r"$x$")
    ax_init_phase.set_ylabel(r"$v$")
    # Plot grid points
    for grid_pt in grid_pts:
        ax_init_phase.axvline(
            (grid_pt * x_scale) + x_min,
            linestyle="--",
            color="k",
            linewidth=0.2,
        )
        ax_init_position.axvline(
            (grid_pt * x_scale) + x_min,
            linestyle="--",
            color="k",
            linewidth=0.2,
        )
    plt.tight_layout()
    save_plot(f"initial_hist_{n}_particles.pdf")
    plt.show()  # Waits for user to close the plot

if "animate_phase_space" in step_flags:
    print("Generating animation of phase space over time.")
    x, v, ke = initialize_state()
    fig2 = plt.figure()

    # fig2, ax = plt.subplots(2, 2, figsize=(12, 10))
    (pt,) = plt.plot([], [], "k.", markersize=1)
    ax.set_ylabel("v")
    ax.set_xlabel("x")
    plt.xlim(x_range)

    # Add a label to the frame showing the current time. Updated each time step
    # in update()
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Evolve positions until t_max. Animate particle positions in phase space.
    animation = FuncAnimation(
        fig2,
        update,
        frames=t_steps,
        init_func=init_animation,
        blit=True,
        interval=10,
        repeat=True,
    )
    plt.show()  # Waits for user to close the plot

if "plot_snapshots" in step_flags:
    print("Generating snapshots of state at various time intervals.")
    snapshot_times = [0, 2 * np.pi, 8 * np.pi]
    t_max = 8 * np.pi
    t_steps = t_steps = math.ceil(t_max / dt)
    x, v, ke = initialize_state()
    snapshot_times.sort()
    snapshot_frames = [math.floor(t / dt) for t in snapshot_times]
    snapshots = []
    print("Simulating...")
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for frame in range(t_steps):
        rho = (
            weight_particles(x, grid_pts, dx, m, order=weighting_order) + rho_bg
        )
        e_j = solve_poisson(rho, inv_a, dx)
        e_i = weight_field(x, grid_pts, e_j, dx, order=weighting_order)
        particle_push(
            x,
            v,
            frame=frame,
            dt=dt,
            e_i=e_i,
            ke=ke,
            x_history=x_history,
            v_history=v_history,
        )
        if frame in snapshot_frames:
            snapshots.append(
                {
                    "x": x_history[:, frame],
                    "v": v_history[:, frame],
                    "frame": frame,
                }
            )
        bar.update(frame + 1)
    bar.finish()
    print(f"Sampled {len(snapshots)} snapshots over {t_steps} time steps.")
    fig3 = plt.figure()
    fig3.suptitle(f"Time snapshots (n={n})")
    num_subplots = len(snapshots)
    idx = 1
    for snapshot in snapshots:
        ax = fig3.add_subplot(num_subplots, 1, idx)
        cur_t = snapshot["frame"] * dt
        ax.set_ylabel("v")
        ax.set_title(f"t={cur_t:.2f}")
        plt.plot(
            (snapshot["x"] * x_scale) + x_min,
            (snapshot["v"] * x_scale),
            "ko",
            markersize=1,
        )
        plt.xlim(x_range)
        if idx == num_subplots:
            ax.set_xlabel("x")
        idx += 1
    plt.tight_layout()
    save_plot(f"snapshots_{n}_particles.pdf")

    plt.show()  # Waits for user to close the plots

if "trace_particles" in step_flags:
    print("Generating trace plots of particles in phase space.")
    t_max = 2 * np.pi
    t_steps = math.ceil(t_max / dt)
    x, v, ke = initialize_state()
    run()
    fig4 = plt.figure()
    fig4.suptitle(f"Particle trajectories (n={n})")
    ax4 = fig4.add_subplot(1, 1, 1)
    for i in range(n):
        position = (x_history[i] * x_scale) + x_min
        velocity = v_history[i] * x_scale
        ax4.plot(position, velocity, "o", markersize=1)
        ax4.set_xlabel("x")
        ax4.set_ylabel("v")
    save_plot(f"traces_{n}_particles.pdf")

    # # Plot total kinetic energy over time
    # fig4_1 = plt.figure()
    # fig4_1.suptitle(f"Total Kinetic Energy (n={n})")
    # ax_ke = fig4_1.add_subplot(1, 1, 1)
    # plt.plot(np.linspace(0, t_max, t_steps), ke * x_scale ** 2)
    # ax_ke.set_ylabel("Total KE")
    # ax_ke.set_xlabel("Time")
    # dt_label = ax_ke.text(
    #     0.02, 0.95, f"Time step: {dt:.4f}", transform=ax_ke.transAxes
    # )
    # save_plot(f"ke_history_{n}_particles_maxt={t_max:.2f}.pdf")
    plt.show()  # Waits for user to close the plots

if "compare_ke" in step_flags:
    print(
        "Generating comparison plots of change in kinetic energy over time for"
        " various time steps."
    )
    # Compare the total kinetic energy over time for various time steps
    t_max = 8 * np.pi
    dt_trials = [0.1, 0.01, 0.001, 0.0001]
    fig5 = plt.figure()
    fig5.suptitle(f"Change in Total Kinetic Energy over Time")
    ax_ke = fig5.add_subplot(1, 3, (1, 2))
    ax_ke.set_ylabel(r"$KE(t)-KE(0)$")
    ax_ke.set_xlabel(r"$t$")
    float_eps = np.finfo(float).eps
    y_max = float_eps
    y_min = -float_eps
    for n in [128, 512, 2048]:
        for dt in dt_trials:
            t_steps = t_steps = math.ceil(t_max / dt)
            x, v, ke = initialize_state()
            run()
            initial_ke = ke[0]
            ke_rel_scaled = (ke - initial_ke) * x_scale ** 2
            if y_max < np.amax(ke_rel_scaled):
                y_max = np.amax(ke_rel_scaled)
            if y_min > np.amin(ke_rel_scaled):
                y_min = np.amin(ke_rel_scaled)
            plt.plot(
                np.linspace(0, t_max, t_steps),
                ke_rel_scaled,
                label=f"n={n}, dt={dt}",
            )
    plt.ylim(y_min, y_max)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)
    save_plot(f"d_ke_comparison.pdf")
    plt.show()  # Waits for user to close the plots


if "performance_testing" in step_flags:
    # Performance testing: print execution time per particle_push(), excluding JIT
    # compilation time

    print("#" * 80 + "\nTesting performance of run():\n" + "#" * 80)
    n = 4098
    dt = 0.05
    x, v, ke = initialize_state()

    # Run one short iteration to compile particle_push()
    t_steps = 1
    x_history = np.zeros((n, t_steps))
    v_history = np.zeros((n, t_steps))
    run()

    # Then we can run the full simulation without counting compile time
    t_max = 8 * np.pi
    t_steps = math.ceil(t_max / dt)
    x, v, ke = initialize_state()

    start_time = time.perf_counter()
    run()
    end_time = time.perf_counter()

    x, v, ke = initialize_state()
    x_history = np.zeros((n, t_steps))
    v_history = np.zeros((n, t_steps))
    start_time_slow = time.perf_counter()
    run(nonumba=True)
    end_time_slow = time.perf_counter()

    print(
        f"(numba ) Total elapsed time per step (n={n}):"
        f" {10**6 * (end_time - start_time) / t_steps:.3f} µs"
    )
    print(
        f"(python) Total elapsed time per step (n={n}):"
        f" {10**6 * (end_time_slow - start_time_slow) / t_steps:.3f} µs"
    )
    print(
        "numba speedup:"
        f" {(end_time_slow - start_time_slow) / (end_time - start_time):.2f}"
        " times faster"
    )

    print("\n" + "#" * 80)
    print("Testing performance of weight_particles(order=0):\n" + "#" * 80)
    m = 32
    grid_pts = np.linspace(0, 1, m)
    dx = 1 / (m - 1)
    # Run one short iteration to compile particle_push()
    weight_particles(x, grid_pts, dx, m, order=0)

    ptime_numba = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x = np.random.uniform(0.0, 1.0, n)
        start_time = time.perf_counter()
        weight_particles(x, grid_pts, dx, m, order=0)
        end_time = time.perf_counter()
        ptime_numba += end_time - start_time
        bar.update(step + 1)
    bar.finish()

    ptime_python = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x = np.random.uniform(0.0, 1.0, n)
        start_time = time.perf_counter()
        weight_particles.py_func(x, grid_pts, dx, m, order=0)
        end_time = time.perf_counter()
        ptime_python += end_time - start_time
        bar.update(step + 1)
    bar.finish()

    print(
        f"(numba ) Total elapsed time per step (n={n}):"
        f" {10**6 * ptime_numba / t_steps:.3f} µs"
    )
    print(
        f"(python) Total elapsed time per step (n={n}):"
        f" {10**6 * ptime_python / t_steps:.3f} µs"
    )
    print(f"numba speedup: {(ptime_python) / (ptime_numba):.2f} times faster")

    print("\n" + "#" * 80)
    print("Testing performance of weight_particles(order=1):\n" + "#" * 80)
    m = 32
    grid_pts = np.linspace(0, 1, m)
    dx = 1 / (m - 1)
    # Run one short iteration to compile particle_push()
    weight_particles(x, grid_pts, dx, m, order=1)

    ptime_numba = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x = np.random.uniform(0.0, 1.0, n)
        start_time = time.perf_counter()
        weight_particles(x, grid_pts, dx, m, order=1)
        end_time = time.perf_counter()
        ptime_numba += end_time - start_time
        bar.update(step + 1)
    bar.finish()

    ptime_python = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x = np.random.uniform(0.0, 1.0, n)
        start_time = time.perf_counter()
        weight_particles.py_func(x, grid_pts, dx, m, order=1)
        end_time = time.perf_counter()
        ptime_python += end_time - start_time
        bar.update(step + 1)
    bar.finish()

    print(
        f"(numba ) Total elapsed time per step (n={n}):"
        f" {10**6 * ptime_numba / t_steps:.3f} µs"
    )
    print(
        f"(python) Total elapsed time per step (n={n}):"
        f" {10**6 * ptime_python / t_steps:.3f} µs"
    )
    print(f"numba speedup: {(ptime_python) / (ptime_numba):.2f} times faster")
