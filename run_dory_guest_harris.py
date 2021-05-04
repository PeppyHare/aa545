"""Dory-Guest-Harris Instability."""
import multiprocessing
import random

import numpy as np

from configuration import Configuration
from model import PicModel
from util import save_data
import plots


class DGHConfiguration(Configuration):
    x_min = -np.pi
    x_max = np.pi
    wp = 1.0
    markersize = 1
    max_history_steps = 2000

    def __init__(
        self,
        v0,
        M=64,
        N=4096,
        k=1,
        wc=1.0,
        wp=1.0,
        n_periods=0.1,
        dt=0.005,
    ):
        self.M = M
        self.N = N
        self.k = k
        self.dt = dt
        self.wc = wc
        self.wp = wp
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
        n_ring = 32  # Number of points per ring in vx/vy
        n_x = int(self.N / n_ring)  # Number of positions to spread rings over
        self.initial_x = np.zeros(self.N)
        self.initial_vx = np.zeros(self.N)
        self.initial_vy = np.zeros(self.N)

        init_positions = np.linspace(self.x_min, self.x_max, n_x + 1)[:-1] + (
            (self.x_max - self.x_min) / n_x / 2
        )
        theta = 2 * np.pi * np.arange(n_ring) / n_ring
        init_positions += 0.0001 * np.cos(self.k * init_positions)
        for group in range(n_x):
            # Disrupt any correlation between x and theta
            np.random.shuffle(theta)
            start_idx = group * n_ring
            end_idx = (group + 1) * n_ring
            self.initial_x[start_idx:end_idx] = init_positions[group]
            self.initial_vx[start_idx:end_idx] = v0 * np.cos(theta)
            self.initial_vy[start_idx:end_idx] = v0 * np.sin(theta)


def calc_dgh(param, wc=10 ** (-1 / 2), wp=1, k=1):
    v0 = param * wc / k
    print(f"k: {k}, wc: {wc:.4f}, v0: {v0:.4f}, wp: {wp:.4f}")
    c = DGHConfiguration(
        v0=v0, n_periods=100, dt=0.01, k=k, M=256, N=8192, wp=wp, wc=wc
    )
    m = PicModel(c)
    plots.plot_initial_distribution(m)
    m.run()
    plots.animate_phase_space(
        m, plot_title="Dory-Guest-Harris Instability", repeat=True, hold=True
    )
    # Save the data!
    save_data(m, f"dgh_{param:.2f}.p")


calc_dgh(6.0, k=1)

# if __name__ == "__main__":
#     param_trials = [4.1, 4.5, 5.0, 5.6, 6.0, 6.6]
#     with multiprocessing.Pool(
#         min(len(param_trials), multiprocessing.cpu_count())
#     ) as p:
#         p.map(calc_dgh, param_trials)
#         p.close()
#     print("phew, that was some work!")