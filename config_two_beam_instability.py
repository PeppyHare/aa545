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
instability.
"""
import numpy as np

###############################################################################
# Settings
###############################################################################
step_flags = [
    # "plot_initial_distributions",
    # "animate_phase_space",
    "plot_snapshots",
    # "plot_dispersion",
    # "no_plots",
    # "trace_particles",
    # "plot_energy",
    "instability_growth_regression",
]

# Number of particles
N = 512

# Number of grid cells. The j=m grid point is identical to j=0.
M = 512

###############################################################################
# Initial Conditions
###############################################################################
#
# Two cold counter-propagating beams with an initial displacement perturbation
# The condition for instability is k^2 v0^2 < wp^2

k = 2
x_min = -np.pi * k
x_max = np.pi * k
v_min = -2
v_max = 2
v0 = 0.1
dx = 0.001
beam1_x = np.linspace(x_min, x_max, int(N / 2 + 1))[:-1]
beam1_x += dx * np.sin(beam1_x)
beam2_x = np.linspace(x_min, x_max, int(N / 2 + 1))[:-1]
beam2_x -= dx * np.sin(beam2_x)
beam1_v = v0 * np.ones_like(beam1_x)
beam2_v = -v0 * np.ones_like(beam2_x)
initial_x = np.concatenate([beam1_x, beam2_x])
initial_v = np.concatenate([beam1_v, beam2_v])

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
# This should be stable
dt = 0.01
# Number of periods of the plasma frequency
n_periods = 20
t_max = n_periods * (2 * np.pi / wp)

# Weighting order. 0 = nearest grid point. 1 = linear weighting
weighting_order = 1

# Should the animation repeat?
repeat_animation = False

# Particle plotting size. 1 is small, 20 is large.
markersize = 2

# Whether to plot grid lines
plot_grid_lines = False

# If plotting snapshots, do so at these values of t
snapshot_times = [0, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max]
print(f"k: {k}, v0: {v0}, wp: {wp}")
# Dispersion relation for opposing, equal-strength streams:
# w = +/- [ k^2 v_0 ^2 + wp ^2 +/- wp*(4 k^2 v0^2 + wp ^2)**(1/2)]**(1/2)
kc = complex(k, 0)
wpc = complex(wp, 0)
v0c = complex(v0, 0)
sol1 = (
    k ** 2 * v0 ** 2
    + wp ** 2
    + wp * (4 * k ** 2 * v0 ** 2 + wp ** 2) ** (1 / 2)
) ** (1 / 2)
sol2 = (
    k ** 2 * v0 ** 2
    + wp ** 2
    - wp * (4 * k ** 2 * v0 ** 2 + wp ** 2) ** (1 / 2)
) ** (1 / 2)
sol3 = -(
    (
        k ** 2 * v0 ** 2
        + wp ** 2
        + wp * (4 * k ** 2 * v0 ** 2 + wp ** 2) ** (1 / 2)
    )
    ** (1 / 2)
)
sol4 = -(
    (
        k ** 2 * v0 ** 2
        + wp ** 2
        - wp * (4 * k ** 2 * v0 ** 2 + wp ** 2) ** (1 / 2)
    )
    ** (1 / 2)
)
print(f"Possible w solutions:\n{sol1:.4f}\n{sol2:.4f}\n{sol3:.4f}\n{sol4:.4f}")
