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


class DGHConfiguration(Configuration):
    x_min = -np.pi
    x_max = np.pi
    wp = 1.0
    markersize = 1
    max_history_steps = 1000

    def __init__(
        self,
        v0,
        M=2048,
        N=4096,
        k=1,
        wc=10 ** (-1 / 2),
        n_periods=0.1,
        dt=0.005,
    ):
        self.M = M
        self.N = N
        self.k = k
        self.dt = dt
        self.wc = wc
        self.v0 = v0
        self.x_min = -np.pi
        self.x_max = np.pi
        self.n_periods = n_periods
        self.weighting_order = 1
        self.vx_min = -1.2 * v0
        self.vx_max = 1.2 * v0
        self.vy_min = -1.2 * v0
        self.vy_max = 1.2 * v0
        Configuration.__init__(self)

    def set_initial_conditions(self):
        v0 = self.v0
        # Try to fill the ring distribution uniformly at each position
        n_groups = int(self.N ** (1/2))
        self.initial_x = np.zeros(self.N)
        self.initial_vx = np.zeros(self.N)
        self.initial_vy = np.zeros(self.N)
        for group in range(n_groups):
            
        self.initial_x = np.linspace(self.x_min, self.x_max, self.N + 1)[:-1]
        theta = 2 * np.pi * np.arange(self.N) / self.N
        # np.random.shuffle(theta)
        self.initial_vx = v0 * np.cos(theta)
        self.initial_vy = v0 * np.sin(theta)


wc = 10 ** (-1 / 2)
k = 1
v0 = 4.5 * wc / k
c = DGHConfiguration(v0=v0, n_periods=1, dt=0.01)
m = PicModel(c)
m.run()
d = m.d
d.fe_hist[0] = d.fe_hist[1]
freq = count_crossings(d.ke_hist) / 4 / (c.n_periods / c.wp)
print(f"Measured frequency: {freq:.2f}, wc: {c.wc:.2f}")
plots.animate_phase_space(m, plot_title="Cyclotron Motion", repeat=True)
