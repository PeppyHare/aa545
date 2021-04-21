#!/usr/bin/env python3
"""
Coding project 1: Electrostatic Particle in Cell.
"""
import math
import os
from pathlib import Path
import time

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
import numpy as np
import progressbar

from weighting import weight_particles, weight_field
from poisson import setup_poisson, compute_field

# There are several pre-configured settings and initial conditions. Include them
# like this:
# from config_langmuir_displacement import (
# from config_langmuir_moving import (
from config_leapfrog_instability import (
    # from config_default import (
    step_flags,
    N,
    M,
    x_min,
    x_max,
    v_min,
    v_max,
    initial_x,
    initial_v,
    eps0,
    qm,
    q,
    rho_bg,
    dt,
    t_max,
    weighting_order,
    repeat_animation,
    markersize,
    snapshot_times,
)

###############################################################################
# Initialization
###############################################################################
x_range = (x_min, x_max)
v_range = (v_min, v_max)

# Scale position from [x_min, x_max] to [0, 1]
# x' = (x - x_min)/(x_max - x_min)
L = x_max - x_min

t_steps = math.ceil(t_max / dt)
frame = 0  # the current time step
time_axis = np.linspace(0, t_max, t_steps)

# Grid spacing / particle size
dx = 1 / M

# Particle mass
m = q / qm

# Initialize arrays
x_i = np.zeros(N)
v_i = np.zeros(N)
e_j = np.zeros(M)
e_max = 0

# Store the history of all particles in phase space
x_hist = np.zeros((N, t_steps))
v_hist = np.zeros((N, t_steps))

# Grid points
x_j = np.linspace(0, 1, M + 1)[:-1]

# Calculate total kinetic energy at each timestep
ke_hist = np.zeros(t_steps)
# Calculate total electric field energy at each timestep
fe_hist = np.zeros(t_steps)
# Calculate total momentum at each timestep
p_hist = np.zeros(t_steps)

# Finite difference matrix used to solve Poisson equation
(inv_a, _) = setup_poisson(M)

# If false, disable live plots of energy
plot_energy = True

# If false, disable live plots of electric field
plot_fields = True


def initialize(x0=initial_x, v0=initial_v):
    """Set initial positions of all particles."""
    global x_i, v_i, t_steps, L, N, M, q, m, qm
    global v_min, v_max, v_stdev
    global x_hist, v_hist, ke_hist, fe_hist, p_hist
    global inv_a, e_i, e_j

    x_i = (x0 - x_min) / L
    v_i = v0 / L

    # Project velocity backwards 1/2 time step
    rho = weight_particles(x_i, x_j, dx, M, q, order=weighting_order) + rho_bg
    # Solve for field at t=0
    e_j = compute_field(rho, inv_a, dx)
    e_i = weight_field(x_i, x_j, e_j, dx, order=weighting_order)
    v_i -= (dt / 2) * qm * e_i

    # (Re-)initialize state history
    x_hist = np.zeros((N, t_steps))
    v_hist = np.zeros((N, t_steps))

    # (Re-)initialize total kinetic energy, field energy, momentum
    ke_hist = np.zeros(t_steps)
    fe_hist = np.zeros(t_steps)
    p_hist = np.zeros(t_steps)

    # Set the number of time steps
    t_steps = math.ceil(t_max / dt)

    return x_i, v_i, ke_hist


###############################################################################
# Time step
###############################################################################
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
    global dx, dt, M, q, qm, m, rho_bg
    global inv_a, weighting_order

    # Particle weighting
    rho = weight_particles(x_i, x_j, dx, M, q, order=weighting_order) + rho_bg

    # Solve Poisson's equation
    e_j = compute_field(rho, inv_a, dx)

    # Calculate total electric field energy
    fe_hist[frame] += dx / 2 * eps0 * np.sum(e_j * e_j)

    # Weight field on grid to particles
    e_i = weight_field(x_i, x_j, e_j, dx, order=weighting_order)

    # Calculate what acceleration will be
    dv = dt * qm * e_i

    # Compute kinetic energy, momentum
    ke_hist[frame] += m / 2 * np.sum(v_i * (v_i + dv))
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
    for i in numba.prange(x_i.size):
        # numba.prange is like np.arange, but optimized for parallelization
        # across multiple CPU cores

        x_i[i] += v_i[i] * dt
        # Apply periodic boundary conditions. NumPy uses the definition of floor
        # where floor(-2.5) == -3.
        x_i[i] -= np.floor(x_i[i])
        x_hist[i][frame] += x_i[i]
        v_hist[i][frame] += v_i[i]
    return x_i, v_i, e_j


def run(nonumba=False):
    """Run the simulation."""
    global x_i, v_i, t_steps, dt, M, dx, x_j
    global ke_hist, fe_hist, p_hist, x_hist, v_hist
    print("Simulating...")
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    initialize()
    for frame in range(t_steps):
        if not nonumba:
            time_step(
                frame,
                x_i=x_i,
                v_i=v_i,
                x_j=x_j,
                ke_hist=ke_hist,
                fe_hist=fe_hist,
                p_hist=p_hist,
                x_hist=x_hist,
                v_hist=v_hist,
                nonumba=nonumba,
            )
        else:
            time_step.pyfunc(
                frame,
                x_i=x_i,
                v_i=v_i,
                x_j=x_j,
                ke_hist=ke_hist,
                fe_hist=fe_hist,
                p_hist=p_hist,
                x_hist=x_hist,
                v_hist=v_hist,
                nonumba=nonumba,
            )
        bar.update(frame)
    bar.finish()
    print("done!")


###############################################################################
# Animation code
###############################################################################
def init_animation():
    """Animation initialization."""
    global ax_xv, pt_xv1, pt_xv2, x_range, v_min, v_max
    ax_xv.set_xlim(x_range)
    ax_xv.set_ylim(v_min, v_max)
    return (pt_xv1, pt_xv2)


def animate(frame):
    """Call every time we update animation frame."""
    global x_i, v_i, dt, ke_hist, L, x_min, x_max
    global pt_ke, pt_fe, pt_te, time_axis, pt_xv1, pt_xv2, time_text, ax_energy
    global ke_hist, fe_hist, p_hist, x_hist, v_hist
    global x_j, e_j, pt_efield, e_max, plot_energy, plot_fields
    if frame == 0:
        x_i, v_i, ke_hist = initialize()
    x_i, v_i, e_j = time_step(
        frame,
        x_i=x_i,
        v_i=v_i,
        x_j=x_j,
        ke_hist=ke_hist,
        fe_hist=fe_hist,
        p_hist=p_hist,
        x_hist=x_hist,
        v_hist=v_hist,
        nonumba=False,
    )
    current_time = frame * dt
    time_text.set_text(f"t = {current_time:.2f}")
    n2 = math.ceil(N / 2)
    pt_xv1.set_data((x_i[:n2] * L) + x_min, (v_i[:n2] * L))
    if n2 > 0:
        pt_xv2.set_data((x_i[n2:] * L) + x_min, (v_i[n2:] * L))
    if plot_fields:
        pt_efield.set_data(
            (np.concatenate([x_j, np.array([x_max])]) * L) + x_min,
            np.concatenate([e_j, e_j[0:1]]),
        )
    if plot_energy:
        pt_ke.set_data(time_axis, ke_hist)
        pt_fe.set_data(time_axis, fe_hist)
        pt_te.set_data(time_axis, fe_hist + ke_hist)
        if fe_hist[frame] + ke_hist[frame] > 1.2 * e_max:
            e_max = fe_hist[frame] + ke_hist[frame]
            ax_energy.set_ylim(0, e_max * 1.4)
    return pt_xv1, pt_xv2, time_text, pt_ke, pt_fe, pt_te, pt_efield


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
    x_i, v_i, ke_hist = initialize()
    bin_width = 0.1

    fig1 = plt.figure()
    fig1.suptitle(f"Initial distribution (n={N})")

    # Plot initial position histogram
    rho_weight_ngp = weight_particles(x_i, x_j, dx, M, order=0)
    print(f"Total charge (ngp): {np.sum(rho_weight_ngp[:-1] * dx)}")
    rho_weight_lin = weight_particles(x_i, x_j, dx, M, order=1)
    print(f"Total charge (linear): {np.sum(rho_weight_lin[:-1] * dx)}")
    ax_init_position = fig1.add_subplot(2, 2, 1)
    bins = math.ceil((x_range[1] - x_range[0]) / bin_width)
    ax_weighted = ax_init_position.twinx()
    ax_init_position.hist((x_i * L) + x_min, bins=bins, range=x_range)
    ax_weighted.step(
        (x_j * L) + x_min,
        rho_weight_ngp,
        color="r",
        marker="o",
        where="mid",
        linewidth=0.5,
    )
    ax_weighted.plot(
        (x_j * L) + x_min,
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
    ax_init_velocity.hist(v_i * L, bins=bins, range=v_range)
    ax_init_velocity.set_xlabel(r"$v$")
    plt.xlim(x_range)
    plt.title("Velocity")

    # Plot initial positions in phase space
    ax_init_phase = fig1.add_subplot(2, 2, (3, 4))
    plt.title("Initial phase space")
    plt.plot((x_i * L) + x_min, (v_i * L), "ko", markersize=1)
    plt.xlim(x_range)
    ax_init_phase.set_xlabel(r"$x$")
    ax_init_phase.set_ylabel(r"$v$")
    # Plot grid points
    for grid_pt in x_j:
        ax_init_phase.axvline(
            (grid_pt * L) + x_min,
            linestyle="--",
            color="k",
            linewidth=0.2,
        )
        ax_init_position.axvline(
            (grid_pt * L) + x_min,
            linestyle="--",
            color="k",
            linewidth=0.2,
        )
    plt.tight_layout()
    save_plot(f"initial_hist_{N}_particles.pdf")
    plt.show()  # Waits for user to close the plot

if "animate_phase_space" in step_flags:
    print("Generating animation of phase space over time.")
    initialize()
    bigfig = plt.figure(figsize=(12, 8))
    bigfig.suptitle(f"n={N}  m={M}  dt={dt:.4f} t_max={t_max:.4f}")
    ax_xv = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
    ax_energy = plt.subplot2grid((2, 2), (1, 0))
    ax_efield = plt.subplot2grid((2, 2), (1, 1))

    ax_xv.set_title("Phase space animation")
    ax_xv.set_ylabel("v")
    ax_xv.set_xlabel("x")
    ax_xv.set_xlim(x_range)
    ax_xv.set_ylim(v_range)

    ax_energy.set_title("Total energy")
    ax_energy.set_xlim(0, t_max)
    ax_energy.set_ylim(0, 2 * (fe_hist[0] + ke_hist[0]))
    ax_energy.set_ylabel("energy")
    ax_energy.set_xlabel("time")

    ax_efield.set_title("Fields")
    ax_efield.set_xlim(x_range)
    ax_efield.set_ylim(v_range)
    ax_efield.set_ylabel("Electric field")
    ax_efield.set_xlabel("x")

    (pt_xv1,) = ax_xv.plot(
        [],
        [],
        "k.",
        color="tab:orange",
        marker=".",
        markersize=markersize,
        label="xv",
    )
    (pt_xv2,) = ax_xv.plot(
        [],
        [],
        "k.",
        color="tab:cyan",
        marker=".",
        markersize=markersize,
        label="xv",
    )
    (pt_ke,) = ax_energy.plot([], [], "b-", markersize=1, label="ke")
    (pt_fe,) = ax_energy.plot([], [], "g-", markersize=1, label="fe")
    (pt_te,) = ax_energy.plot([], [], "k-", markersize=1, label="total")
    ax_energy.legend(
        [pt_ke, pt_fe, pt_te],
        [pt_ke.get_label(), pt_fe.get_label(), pt_te.get_label()],
    )
    (pt_efield,) = ax_efield.plot([], [], "c", label="efield")

    # Add the grid points to the plot of the fields, but only if there aren't
    # too many of them
    if x_j.size < 80:
        for grid_pt in x_j:
            ax_efield.axvline(
                (grid_pt * L) + x_min,
                linestyle="--",
                color="k",
                linewidth=0.2,
            )

    # Add a label to the frame showing the current time. Updated each time step
    # in update()
    time_text = ax_xv.text(0.02, 0.95, "", transform=ax_xv.transAxes)

    # Evolve positions until t_max. Animate particle positions in phase space.
    animation = FuncAnimation(
        bigfig,
        animate,
        frames=t_steps,
        init_func=init_animation,
        blit=True,
        interval=1,
        repeat=repeat_animation,
    )
    plt.tight_layout()
    plt.show()  # Waits for user to close the plot

if "plot_snapshots" in step_flags:
    print("Generating snapshots of state at various time intervals.")
    t_steps = t_steps = math.ceil(t_max / dt)
    initialize()
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
        rho = weight_particles(x_i, x_j, dx, M, order=weighting_order) + rho_bg
        e_j = compute_field(rho, inv_a, dx)
        e_i = weight_field(x_i, x_j, e_j, dx, order=weighting_order)
        time_step(
            frame,
            x_i=x_i,
            v_i=v_i,
            x_j=x_j,
            ke_hist=ke_hist,
            fe_hist=fe_hist,
            p_hist=p_hist,
            x_hist=x_hist,
            v_hist=v_hist,
            nonumba=False,
        )
        if frame in snapshot_frames:
            snapshots.append(
                {
                    "x": x_hist[:, frame],
                    "v": v_hist[:, frame],
                    "frame": frame,
                }
            )
        bar.update(frame + 1)
    bar.finish()
    print(f"Sampled {len(snapshots)} snapshots over {t_steps} time steps.")
    fig3 = plt.figure()
    fig3.suptitle(f"Time snapshots (n={N})")
    num_subplots = len(snapshots)
    idx = 1
    for snapshot in snapshots:
        ax_xv = fig3.add_subplot(num_subplots, 1, idx)
        cur_t = snapshot["frame"] * dt
        ax_xv.set_ylabel("v")
        ax_xv.set_title(f"t={cur_t:.2f}")
        plt.plot(
            (snapshot["x"] * L) + x_min,
            (snapshot["v"] * L),
            "ko",
            markersize=1,
        )
        plt.xlim(x_range)
        if idx == num_subplots:
            ax_xv.set_xlabel("x")
        idx += 1
    plt.tight_layout()
    save_plot(f"snapshots_{N}_particles.pdf")

    plt.show()  # Waits for user to close the plots

if "trace_particles" in step_flags:
    print("Generating trace plots of particles in phase space.")
    t_max = 2 * np.pi
    t_steps = math.ceil(t_max / dt)
    initialize()
    run()
    fig4 = plt.figure()
    fig4.suptitle(f"Particle trajectories (n={N})")
    ax4 = fig4.add_subplot(1, 1, 1)
    for i in range(N):
        position = (x_hist[i] * L) + x_min
        velocity = v_hist[i] * L
        ax4.plot(position, velocity, "o", markersize=1)
        ax4.set_xlabel("x")
        ax4.set_ylabel("v")
    save_plot(f"traces_{N}_particles.pdf")

    plt.show()  # Waits for user to close the plots

if "compare_ke" in step_flags:
    print(
        "Generating comparison plots of change in kinetic energy over time for"
        " various time steps."
    )
    # Compare the total kinetic energy over time for various time steps
    t_max = 8 * np.pi
    dt_trials = [0.1, 0.01, 0.001]
    fig5 = plt.figure()
    fig5.suptitle("Change in Total Kinetic Energy over Time")
    ax_ke = fig5.add_subplot(1, 3, (1, 2))
    ax_ke.set_ylabel(r"$KE(t)-KE(0)$")
    ax_ke.set_xlabel(r"$t$")
    float_eps = np.finfo(float).eps
    y_max = float_eps
    y_min = -float_eps
    for npart in [128, 512, 2048]:
        N = npart
        for trial in dt_trials:
            dt = trial
            t_steps = t_steps = math.ceil(t_max / dt)
            x_i, v_i, ke_hist = initialize()
            run()
            initial_ke = ke_hist[0]
            ke_rel_scaled = (ke_hist - initial_ke) * L ** 2
            if y_max < np.amax(ke_rel_scaled):
                y_max = np.amax(ke_rel_scaled)
            if y_min > np.amin(ke_rel_scaled):
                y_min = np.amin(ke_rel_scaled)
            plt.plot(
                np.linspace(0, t_max, t_steps),
                ke_rel_scaled,
                label=f"n={N}, dt={dt}",
            )
    plt.ylim(y_min, y_max)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)
    save_plot("d_ke_comparison.pdf")
    plt.show()  # Waits for user to close the plots


if "performance_testing" in step_flags:
    # Performance testing: print execution time per particle_push(), excluding
    # JIT compilation time

    print("#" * 80 + "\nTesting performance of run():\n" + "#" * 80)
    N = 4098
    dt = 0.05

    # Run one short iteration to compile particle_push()
    t_steps = 1
    initialize()
    x_hist = np.zeros((N, t_steps))
    v_hist = np.zeros((N, t_steps))
    run()

    # Then we can run the full simulation without counting compile time
    t_max = 8 * np.pi
    t_steps = math.ceil(t_max / dt)
    initialize()

    start_time = time.perf_counter()
    run()
    end_time = time.perf_counter()

    initialize()
    x_hist = np.zeros((N, t_steps))
    v_hist = np.zeros((N, t_steps))
    start_time_slow = time.perf_counter()
    run(nonumba=True)
    end_time_slow = time.perf_counter()

    print(
        f"(numba ) Total elapsed time per step (n={N}):"
        f" {10**6 * (end_time - start_time) / t_steps:.3f} µs"
    )
    print(
        f"(python) Total elapsed time per step (n={N}):"
        f" {10**6 * (end_time_slow - start_time_slow) / t_steps:.3f} µs"
    )
    print(
        "numba speedup:"
        f" {(end_time_slow - start_time_slow) / (end_time - start_time):.2f}"
        " times faster"
    )

    print("\n" + "#" * 80)
    print("Testing performance of weight_particles(order=0):\n" + "#" * 80)
    M = 32
    x_j = np.linspace(0, 1, M)
    dx = 1 / (M - 1)
    # Run one short iteration to compile particle_push()
    weight_particles(x_i, x_j, dx, M, order=0)

    ptime_numba = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x_i = np.random.uniform(0.0, 1.0, N)
        start_time = time.perf_counter()
        weight_particles(x_i, x_j, dx, M, order=0)
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
        x_i = np.random.uniform(0.0, 1.0, N)
        start_time = time.perf_counter()
        weight_particles.py_func(x_i, x_j, dx, M, order=0)
        end_time = time.perf_counter()
        ptime_python += end_time - start_time
        bar.update(step + 1)
    bar.finish()

    print(
        f"(numba ) Total elapsed time per step (n={N}):"
        f" {10**6 * ptime_numba / t_steps:.3f} µs"
    )
    print(
        f"(python) Total elapsed time per step (n={N}):"
        f" {10**6 * ptime_python / t_steps:.3f} µs"
    )
    print(f"numba speedup: {(ptime_python) / (ptime_numba):.2f} times faster")

    print("\n" + "#" * 80)
    print("Testing performance of weight_particles(order=1):\n" + "#" * 80)
    M = 32
    x_j = np.linspace(0, 1, M)
    dx = 1 / (M - 1)
    # Run one short iteration to compile particle_push()
    weight_particles(x_i, x_j, dx, M, order=1)

    ptime_numba = 0
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for step in range(t_steps):
        x_i = np.random.uniform(0.0, 1.0, N)
        start_time = time.perf_counter()
        weight_particles(x_i, x_j, dx, M, order=1)
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
        x_i = np.random.uniform(0.0, 1.0, N)
        start_time = time.perf_counter()
        weight_particles.py_func(x_i, x_j, dx, M, order=1)
        end_time = time.perf_counter()
        ptime_python += end_time - start_time
        bar.update(step + 1)
    bar.finish()

    print(
        f"(numba ) Total elapsed time per step (n={N}):"
        f" {10**6 * ptime_numba / t_steps:.3f} µs"
    )
    print(
        f"(python) Total elapsed time per step (n={N}):"
        f" {10**6 * ptime_python / t_steps:.3f} µs"
    )
    print(f"numba speedup: {(ptime_python) / (ptime_numba):.2f} times faster")
