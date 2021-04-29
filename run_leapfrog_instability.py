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

from configuration import Configuration
from model import PicModel
from util import count_crossings
import plots


class LeapfrogInstabilityConfiguration(Configuration):
    N = 64
    M = 32
    x_min = -np.pi
    x_max = np.pi
    v_min = -0.5
    v_max = 0.5
    wp = 1.0
    dt = 1.0 / (2 / wp)
    n_periods = 8

    def initialize_particles(self):
        initial_x = np.linspace(self.x_min, self.x_max, self.N + 1)[:-1]
        initial_v = 0.001 * np.sin(initial_x)
        # Split odd/even points by colors for more useful plotting
        initial_x = np.concatenate([initial_x[::2], initial_x[1::2]])
        initial_v = np.concatenate([initial_v[::2], initial_v[1::2]])

        self.initial_x = initial_x
        self.initial_v = initial_v


c = LeapfrogInstabilityConfiguration()
m = PicModel(c)
plots.plot_initial_distribution(m)
m.run()
# Get the data from the run
d = m.d
plots.animate_phase_space(m, plot_title="Leapfrog Instability demo")
print(f"dt: {c.dt:.2f}")
print(f"wp: {c.wp:.2f}")

# Try to determine the observed frequency of oscillations
print(f"crossings(ke): {count_crossings(d.ke_hist)}")
print(f"crossings(x[0]): {count_crossings(d.x_hist[0,:])}")
print(f"t_max/2π: {c.t_max/(2*np.pi)}")
print(f"dt*wp/2: {c.dt*c.wp/2:.4f}")
print(
    "measured w*dt/2:"
    f" {c.dt*count_crossings(d.ke_hist)/(4*c.t_max/(2*np.pi)):.4f}"
)
