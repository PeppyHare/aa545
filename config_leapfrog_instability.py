"""Leapfrog dispersion of Langmuir oscillations.

Use only 1st order weighting for this part. Initialize 64 cold, static, evenly
distributed particles onto a grid with 32 cells (33 grid points). Initialize a
small velocity perturbation that varies sinusoidally with a single period in x.
The velocity perturbation should be small enough to maintain harmonic
oscillations. (The particle trajectories should not cross.) Measure the
oscillation frequency from the energy history plot and compare it to the plasma
frequency. Repeat for several plasma frequency values and plot the deviation of
the measured frequency from the plasma frequency. Compare with the theoretical
result for the leap-frog instability to demonstrate phase error and
instability."""
import numpy as np

###############################################################################
# Settings
###############################################################################
step_flags = [
    # "plot_initial_distributions",
    "animate_phase_space",
    # "plot_snapshots",
    # "trace_particles",
    # "compare_ke",
    # "performance_testing",
]

# Number of particles
N = 64

# Number of grid cells. The j=m grid point is identical to j=0.
M = 32

###############################################################################
# Initial Conditions
###############################################################################
#
# Two stationary particles separated by L/4
#

x_min = -np.pi
x_max = np.pi
v_min = -0.5
v_max = 0.5
initial_x = np.linspace(x_min, x_max, N)
# initial_v = 0.01 * np.sin(4*initial_x)
initial_v = np.zeros(N)
# Split odd/even points by colors for more useful plotting
initial_x = np.concatenate([initial_x[::2], initial_x[1::2]])
initial_v = np.concatenate([initial_v[::2], initial_v[1::2]])

# Plasma electron frequency
wp = 1
# Normalize epsilon_0
eps0 = 1
# Charge-to-mass ratio of species
qm = -1
# Particle charge
q = (wp ** 2) * eps0 / (N * qm)
# Background charge density
rho_bg = -N * q

# Time step and duration
dt = 0.005
t_max = 8 * np.pi

# Weighting order. 0 = nearest grid point. 1 = linear weighting
weighting_order = 1

# Should the animation repeat?
repeat_animation = False

# Particle plotting size. 1 is small, 20 is large.
markersize = 10

# If plotting snapshots, do so at these values of t
snapshot_times = [0, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max]
