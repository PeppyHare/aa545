"""
Write a code that evolves free streaming particles in phase space, (x, vx). Assume there are no forces and no collisions. Track the position and velocity of each particle. The code should work with an arbitrary number of particles. Implement periodic boundary conditions in the x direction. Visualize the motion of the particles by plotting their positions in phase space as a function of time. Also, plot particle trajectories by not erasing their positions from the previous time. Implement a diagnostic that computes the time history of the kinetic energy of all particles.

Initialize a randomly filled particle distribution that is Maxwellian in vx and is uniform in x using N = 128, 512, 2048. The FWHM in velocity should be 2. Use a domain that is vx = [-5, 5] and x = [-2π, 2π]. For the three cases, generate a histogram of the particle density as a function of vx, using a bin width of 0.25.
Evolve the particle motion until t = 8π.
Produce plots of the particle positions for the N = 512 case at t = 0, 2π, & 8π. Show the plot of the particle trajectories for the N = 128 case up to t = 2π.
Compare plots of the kinetic energy history for all four cases for different time step values.
Upload your report in pdf format, source code, and run script/instructions.
"""
import random
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from numba import jit


# n = 128
n = 16384
dt = 0.05
t_max = 8 * np.pi

fwhm = 2.0
x_range = (-2 * np.pi, 2 * np.pi)
v_range = (-5.0, 5.0)
# The FWHM in velocity should be 2.
# FWHM = 2*Sqrt(2*ln(2))σ ~ 2.355σ
stdev = fwhm / 2.355


@jit(nopython=True)  # numba makes it fast!
def time_step(state, dt):
    """Evolve state forward in time by dt with periodic boundary conditions."""
    # do this the dummy way for now
    for i in np.arange(n):
        # No collisions or forces
        state[0][i] += state[1][i] * dt

        # Apply boundary conditions
        if state[0][i] > 2 * np.pi:
            state[0][i] -= 4 * np.pi
        if state[0][i] < -2 * np.pi:
            state[0][i] += +4 * np.pi
    return state


initial_state = np.zeros((2, n))
for i in range(n):
    # Initialize a randomly filled particle distribution that is Maxwellian in vx and is uniform in x.
    initial_state[0][i] = random.uniform(-2 * np.pi, 2 * np.pi)
    initial_state[1][i] = max(min(random.gauss(0, stdev), 5), -5)

current_state = initial_state
fig, ax = plt.subplots(figsize=(12, 10))
(pt,) = plt.plot([], [], "r.", markersize=0.3)
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)


def init():
    ax.set_xlim(x_range)
    ax.set_ylim(v_range[0]/2, v_range[1]/2)
    return (pt,)


def update(frame):
    time_step(current_state, dt=dt)
    current_time = frame * dt
    time_text.set_text(f"t = {current_time:.2f}")
    pt.set_data(current_state[0], current_state[1])
    return tuple([pt]) + tuple([time_text])


ani = FuncAnimation(fig, update, frames=math.ceil(t_max / dt), init_func=init, blit=True, interval=1)
plt.show()


def main():
    print(initial_state)
    # plt.subplot(2, 2, 1)
    # bins = math.ceil((x_range[1] - x_range[0]) / 0.025)
    # plt.hist(initial_state[0], bins=bins, range=x_range)
    # plt.title("Initial position distribution")
    # plt.subplot(2, 2, 2)
    # bins = math.ceil((v_range[1] - v_range[0]) / 0.025)
    # plt.hist(initial_state[1], bins=bins, range=v_range)
    # plt.title("Initial velocity distribution")
    # plt.subplot(2, 2, (3, 4))
    # plt.title("Phase space distribution")
    # plt.plot(state[0], state[1], "r.")
    # plt.show()
