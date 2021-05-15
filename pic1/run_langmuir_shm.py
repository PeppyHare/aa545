"""Simple harmonic motion of Langmuir oscillations.

Initialize only two particles of equal charge, separated by a small
displacement. Initial velocity of each particle is zero. Once released, the
particles will undergo simple harmonic motion.
"""
import numpy as np

from configuration import Configuration
from model import PicModel
import plots
from util import count_crossings


class SHMConfiguration(Configuration):
    N = 2
    M = 32
    x_min = -3 * np.pi
    x_max = 3 * np.pi
    vx_min = -1
    vx_max = 1
    wp = 1.0
    dt = 0.0001
    n_periods = 8
    markersize = 10
    max_history_steps = 1000

    def set_initial_conditions(self):
        initial_x = [-np.pi / 4, np.pi / 4]
        initial_vx = np.zeros_like(initial_x)
        # Split odd/even points by colors for more useful plotting
        initial_x = np.concatenate([initial_x[::2], initial_x[1::2]])
        initial_vx = np.concatenate([initial_vx[::2], initial_vx[1::2]])

        self.initial_x = initial_x
        self.initial_vx = initial_vx
        self.initial_vy = np.zeros_like(initial_vx)


c = SHMConfiguration()
m = PicModel(c)
# plots.plot_initial_distribution(m)
m.run()
# Get the data from the run
d = m.d
freq = count_crossings(d.fe_hist) / 4 / c.n_periods
print(f"Measured frequency: {freq:.2f}, wp: {c.wp:.2f}")
plots.animate_phase_space(
    m, plot_title="2-particle simple harmonic motion", repeat=True
)
