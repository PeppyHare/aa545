"""Langmuir oscillations with bulk velocity.

Initialize two particles that are evenly distributed on a spatially periodic
domain, x = [-π,π] with 32 cells, with nonzero initial velocities, i.e. the
phase space position for the two particles are (−π/2,v') and (π/2,v'). Adjust
v' to generate simple harmonic motion.  For the 1st order weighting scheme only,
plot the energy histories and measure the oscillation frequency. Compare to the
theoretically expected value for both 0th and 1st. Increase v' so the particles’
trajectories cross."""
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
N = 2

# Number of grid cells. The j=m grid point is identical to j=0.
M = 320

###############################################################################
# Initial Conditions
###############################################################################
#
# Two stationary particles separated by L/4
#

x_min = -np.pi
x_max = np.pi
v_min = -5
v_max = 5
v_bulk = 2
initial_x = np.zeros(N)
initial_v = np.zeros(N)
initial_x[0] = -np.pi / 2
initial_x[1] = np.pi / 2
initial_v[0] = v_bulk
initial_v[1] = -v_bulk

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
dt = 0.05
t_max = 8 * np.pi

# Weighting order. 0 = nearest grid point. 1 = linear weighting
weighting_order = 1

# Should the animation repeat?
repeat_animation = False

# Particle plotting size. 1 is small, 20 is large.
markersize = 20

# If plotting snapshots, do so at these values of t
snapshot_times = [0, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max]
