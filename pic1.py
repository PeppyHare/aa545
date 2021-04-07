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
import random
import math
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit
import progressbar


###############################################################################
# Settings
###############################################################################

# Plot histograms of the initial position and velocity and plot the initial
# state in phase space
plot_initial_distributions = True

# Create an animation of phase space over time
# Mutually exclusive with generating timeseries plots
animate_phase_space = True

# Plot snapshots of phase space at various times
plot_snapshots = True

# Plot traces of particles in phase space over time
trace_particles = True

# Run simulation using various values for the time step and plot the change in
# the total kinetic energy diagnostic over time
compare_ke = True

# Number of particles
n = 128
# n = 512
# n = 2048

# Initial position distribution: uniform over [-2π, 2π]
x_min = -2 * np.pi
x_max = 2 * np.pi
# Initial velocity distribution: Maxwellian with FWHM=2
v_fwhm = 2.0
v_min = -5.0
v_max = 5.0

# Time step and duration
dt = 0.01
t_max = 8 * np.pi


###############################################################################
# Initialization
###############################################################################
x_range = (x_min, x_max)
v_range = (v_min, v_max)
# By definition, FWHM = 2*Sqrt(2*ln(2))σ ~ 2.355σ
v_stdev = v_fwhm / 2.355

t_steps = math.ceil(t_max / dt)
frame = 0  # the current time step

# Initialize arrays
current_state = np.zeros((2, n))

# Store the history of all particles
history = np.zeros((2, n, t_steps))

# Calculate the kinetic energy at each timestep
ke = np.zeros(t_steps)


def initialize_state():
    """Set initial positions of all particles."""
    global current_state, history, ke, x_min, x_max, v_min, v_max, v_stdev
    # Fixing random state for reproducibility
    random.seed("not really random")
    initial_state = np.zeros((2, n))
    for i in range(n):
        initial_state[0][i] = random.uniform(x_min, x_max)
        initial_state[1][i] = max(min(random.gauss(0, v_stdev), v_max), v_min)
    current_state = initial_state

    # (Re-)initialize state history
    history = np.zeros((2, n, t_steps))

    # (Re-)initialize total kinetic energy
    ke = np.zeros(t_steps)

    return initial_state


###############################################################################
# Time step
###############################################################################
@jit(nopython=True, cache=True)  # numba makes it fast!
def time_step(state, frame, dt, history, ke):
    """Evolve state forward in time by dt with periodic boundary conditions."""
    for i in np.arange(n):
        # No collisions or forces, just dx=v*dt
        state[0][i] += state[1][i] * dt

        # Apply boundary conditions
        if state[0][i] > 2 * np.pi:
            state[0][i] -= 4 * np.pi
        if state[0][i] < -2 * np.pi:
            state[0][i] += 4 * np.pi

        history[0][i][frame] += state[0][i]
        history[1][i][frame] += state[1][i]

    ke[frame] += 0.5 * (np.sum(np.square(state[1])))
    frame += 1
    return state


def run():
    """Run the simulation."""
    global current_state, t_steps, dt, history, ke
    print("Simulating...")
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for frame in range(t_steps):
        time_step(current_state, frame=frame, dt=dt, history=history, ke=ke)
        bar.update(frame + 1)
    bar.finish()
    print("done!")


###############################################################################
# Animation code
###############################################################################
def init():
    """Animation initialization."""
    global ax, x_range, v_min, v_max
    ax.set_xlim(x_range)
    ax.set_ylim(v_min, v_max)
    return (pt,)


def update(frame):
    """Call every time we update animation frame."""
    global current_state, dt, time_text, pt
    time_step(current_state, frame=frame, dt=dt, history=history, ke=ke)
    current_time = frame * dt
    time_text.set_text(f"t = {current_time:.2f}")
    pt.set_data(current_state[0], current_state[1])
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
if plot_initial_distributions:
    print("Generating plots of initial particle state.")
    initial_state = initialize_state()
    bin_width = 0.25

    fig1 = plt.figure()
    fig1.suptitle(f"Initial distribution (n={n})")

    # Plot initial position histogram
    ax_init_position = fig1.add_subplot(2, 2, 1)
    bins = math.ceil((x_range[1] - x_range[0]) / bin_width)
    ax_init_position.hist(initial_state[0], bins=bins, range=x_range)
    ax_init_position.set_xlabel(r"$x$")
    ax_init_position.set_ylabel("Count")
    plt.title("Position")

    # Plot initial velocity histogram
    ax_init_velocity = fig1.add_subplot(2, 2, 2)
    bins = math.ceil((v_range[1] - v_range[0]) / bin_width)
    ax_init_velocity.hist(initial_state[1], bins=bins, range=v_range)
    ax_init_velocity.set_xlabel(r"$v$")
    plt.title("Velocity")

    # Plot initial positions in phase space
    ax_init_phase = fig1.add_subplot(2, 2, (3, 4))
    plt.title("Initial phase space")
    plt.plot(current_state[0], current_state[1], "ko", markersize=1)
    ax_init_phase.set_xlabel(r"$x$")
    ax_init_phase.set_ylabel(r"$v$")
    plt.tight_layout()
    save_plot(f"initial_hist_{n}_particles.pdf")
    plt.show()  # Waits for user to close the plot

if animate_phase_space:
    print("Generating animation of phase space over time.")
    fig2, ax = plt.subplots(figsize=(12, 10))
    (pt,) = plt.plot([], [], "k.", markersize=1)

    # Add a label to the frame showing the current time. Updated each time step
    # in update()
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Evolve positions until t_max. Animate particle positions in phase space.
    animation = FuncAnimation(
        fig2, update, frames=t_steps, init_func=init, blit=True, interval=1
    )
    plt.show()  # Waits for user to close the plot

if plot_snapshots:
    print("Generating snapshots of state at various time intervals.")
    initialize_state()
    run()

    # Plot phase space snapshots
    fig3 = plt.figure()
    fig3.suptitle(f"Time Snapshots (n={n})")
    ax_t0 = fig3.add_subplot(3, 1, 1)
    ax_t0.set_ylabel("v")
    plt.plot(history[0, ..., 0], history[1, ..., 0], "ko", markersize=1)

    ax_t1 = fig3.add_subplot(3, 1, 2)
    ax_t1.set_ylabel("v")
    t1 = 2.0 * np.pi
    frame_t1 = int(t1 / dt)
    plt.plot(
        history[0, ..., frame_t1], history[1, ..., frame_t1], "ko", markersize=1
    )

    ax_t2 = fig3.add_subplot(3, 1, 3)
    ax_t2.set_ylabel("v")
    ax_t0.set_xlabel("x")
    t2 = 8.0 * np.pi
    frame_t2 = int(t2 / dt)
    plt.plot(
        history[0, ..., frame_t2], history[1, ..., frame_t2], "ko", markersize=1
    )
    plt.tight_layout()
    save_plot(f"snapshots_{n}_particles.pdf")

    # Plot kinetic energy
    fig3_1 = plt.figure()
    fig3_1.suptitle(f"Total Kinetic Energy (n={n})")
    ax_ke = fig3_1.add_subplot(1, 1, 1)
    plt.plot(np.linspace(0, t_max, t_steps), ke)
    ax_ke.set_ylabel("Total KE")
    ax_ke.set_xlabel("Time")

    plt.show()  # Waits for user to close the plots

if trace_particles:
    print("Generating trace plots of particles in phase space.")
    t_max = 2 * np.pi
    t_steps = t_steps = math.ceil(t_max / dt)
    initialize_state()
    print("Simulating...")
    for frame in range(t_steps):
        time_step(current_state, frame=frame, dt=dt, history=history, ke=ke)
    print("done!")
    # try just plotting the first particle's trajectory
    fig4 = plt.figure()
    fig4.suptitle(f"Particle trajectories (n={n})")
    ax4 = fig4.add_subplot(1, 1, 1)
    for i in range(n):
        position = history[0][i]
        velocity = history[1][i]
        ax4.plot(position, velocity, "o", markersize=1)
        ax4.set_xlabel("x")
        ax4.set_ylabel("v")
    save_plot(f"traces_{n}_particles.pdf")

    # Plot total kinetic energy over time
    fig4_1 = plt.figure()
    fig4_1.suptitle(f"Total Kinetic Energy (n={n})")
    ax_ke = fig4_1.add_subplot(1, 1, 1)
    plt.plot(np.linspace(0, t_max, t_steps), ke)
    ax_ke.set_ylabel("Total KE")
    ax_ke.set_xlabel("Time")
    dt_label = ax_ke.text(
        0.02, 0.95, f"Time step: {dt:.4f}", transform=ax_ke.transAxes
    )
    save_plot(f"ke_history_{n}_particles_maxt={t_max:.2f}.pdf")
    plt.show()  # Waits for user to close the plots

if compare_ke:
    print(
        "Generating comparison plots of change in kinetic energy over time for various time steps."
    )
    # Compare the total kinetic energy over time for various time steps
    t_max = 8 * np.pi
    dt_trials = [0.1, 0.01, 0.001, 0.0001]
    fig5 = plt.figure()
    fig5.suptitle(f"Change in Total Kinetic Energy (n={n})")
    ax_ke = fig5.add_subplot(1, 1, 1)
    ax_ke.set_ylabel(r"$KE(t)-KE(0)$")
    for dt in dt_trials:
        t_steps = t_steps = math.ceil(t_max / dt)
        initial_state = initialize_state()
        initial_ke = 0.5 * (np.sum(np.square(initial_state[1])))
        run()
        plt.plot(
            np.linspace(0, t_max, t_steps), (ke - initial_ke), label=f"dt={dt}"
        )
    ax_ke.legend()
    save_plot(f"d_ke_{n}_particles.pdf")
    plt.show()  # Waits for user to close the plots
