"""Simple particle undergoing cyclotron motion.

Initializes a single particle with an initial velocity in the y-direction in a
static magnetic field. Particle begins centered on a spatially periodic domain,
and has no initial velocity in the x-direction.
"""
import numpy as np

from configuration import Configuration
from model import PicModel
import plots
from util import count_crossings


class CyclotronConfiguration(Configuration):
    N = 1
    M = 64
    x_min = -np.pi
    x_max = np.pi
    vx_min = -1.2
    vx_max = 1.2
    vy_min = -1.2
    vy_max = 1.2
    wp = 1.0
    wc = 1.0
    dt = 0.05
    n_periods = 2
    markersize = 10
    max_history_steps = 1000

    def set_initial_conditions(self):
        self.initial_x = -1 * np.ones(1)
        self.initial_vx = np.zeros(1)
        self.initial_vy = np.ones(1)


c = CyclotronConfiguration()
m = PicModel(c)
m.run()
d = m.d
freq = count_crossings(d.x_hist[0]) / 2 / (c.n_periods / c.wp)
print(f"Measured frequency: {freq:.2f}, wc: {c.wc:.2f}")
# plots.animate_phase_space(m, plot_title="Cyclotron Motion", repeat=True)
plots.plot_traces(m, plot_title=r"Cyclotron Motion ($\omega_c = 1$)")
