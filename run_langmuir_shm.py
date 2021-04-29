"""Simple harmonic motion of Langmuir oscillations.

Initialize only two particles of equal charge, separated by a small
displacement. Initial velocity of each particle is zero. Once released, the
particles will undergo simple harmonic motion.
"""
import numpy as np

from configuration import Configuration
from model import PicModel
import plots


class SHMConfiguration(Configuration):
    N = 2
    M = 32
    x_min = -np.pi
    x_max = np.pi
    v_min = -2
    v_max = 2
    wp = 1.0
    dt = 0.01
    n_periods = 8
    markersize = 10

    def initialize_particles(self):
        initial_x = [-np.pi / 4, np.pi / 4]
        initial_v = np.zeros_like(initial_x)
        # Split odd/even points by colors for more useful plotting
        initial_x = np.concatenate([initial_x[::2], initial_x[1::2]])
        initial_v = np.concatenate([initial_v[::2], initial_v[1::2]])

        self.initial_x = initial_x
        self.initial_v = initial_v


c = SHMConfiguration()
m = PicModel(c)
plots.plot_initial_distribution(m)
m.run()
# Get the data from the run
d = m.d
plots.animate_phase_space(m, plot_title="Leapfrog Instability demo")
