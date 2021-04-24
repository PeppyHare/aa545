"""Global default configuration settings for electrostatic PIC solver"""
import random
import numpy as np

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
    "plot_initial_distributions",
    "animate_phase_space",
    "plot_snapshots",
    "trace_particles",
    "compare_ke",
]

# Number of particles
N = 128

# Number of grid cells. The j=m grid point is identical to j=0.
M = 32

###############################################################################
# Initial Conditions
###############################################################################

# Initial position distribution: uniform over [-2π, 2π]
x_min = -2 * np.pi
x_max = 2 * np.pi
# Initial velocity distribution: Maxwellian with FWHM=2
v_fwhm = 2
v_min = -5
v_max = 5
# For normal distribution, FWHM = 2*Sqrt(2*ln(2))σ ~ 2.355σ
v_stdev = v_fwhm / 2.355
random.seed("not really random")
initial_x = np.zeros(N)
initial_v = np.zeros(N)
for i in range(N):
    initial_x[i] = random.uniform(x_min, x_max)
    initial_v[i] = max(min(random.gauss(0, v_stdev), v_max), v_min)

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
repeat_animation = True

# Particle plotting size. 1 is small, 20 is large.
markersize = 3

# If plotting snapshots, do so at these values of t
snapshot_times = [0, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max]
