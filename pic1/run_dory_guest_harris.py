"""Dory-Guest-Harris Instability."""
import os

import numpy as np
from matplotlib import pyplot as plt

from pic1.configuration import Configuration
from pic1.model import PicModel
from pic1.util import load_data, save_data
import pic1.plots as plots

plt.style.use("dark_background")
demo_mode = True  # Don't re-run simulation during a demo


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
        n_ring = 128  # Number of points per ring in vx/vy
        n_x = int(self.N / n_ring)  # Number of positions to spread rings over
        self.initial_x = np.zeros(self.N)
        self.initial_vx = np.zeros(self.N)
        self.initial_vy = np.zeros(self.N)

        init_positions = np.linspace(self.x_min, self.x_max, n_x + 1)[:-1] + (
            (self.x_max - self.x_min) / n_x / 2
        )
        theta = 2 * np.pi * np.arange(n_ring) / n_ring
        init_positions += 0.001 * np.cos(self.k * init_positions)
        for group in range(n_x):
            # Disrupt any correlation between x and theta. Only needed if
            # initial velocity distribution is warm
            # np.random.shuffle(theta)
            start_idx = group * n_ring
            end_idx = (group + 1) * n_ring
            self.initial_x[start_idx:end_idx] = init_positions[group]
            self.initial_vx[start_idx:end_idx] = v0 * np.cos(theta)
            self.initial_vy[start_idx:end_idx] = v0 * np.sin(theta)


def run_dgh(param, wc=10 ** (-1 / 2), wp=1, k=1, n_periods=30):
    """Run the PIC code with a cold ring distribution initialized.

    The :param: parameter is k*v/wc, which determines the stability of the
    solutions."""
    v0 = param * wc / k
    print(f"k: {k}, wc: {wc:.4f}, v0: {v0:.4f}, wp: {wp:.4f}")
    print("Setting up initial particle configuration.")
    c = DGHConfiguration(
        v0=v0, n_periods=n_periods, dt=0.003, k=k, M=64, N=8192, wp=wp, wc=wc
    )
    print("Initializing model and compiling subroutines.")
    m = PicModel(c)
    plots.plot_initial_distribution(m)
    print(f"Time steps: {c.t_steps}")
    m.run()
    plots.animate_phase_space(
        m,
        plot_title=(
            f"Dory-Guest-Harris Instability: $k v_0 / \omega_c = {param:.2f}$"
        ),
        repeat=True,
        hold=True,
    )
    # Save the data!
    save_data(m, f"dgh_{param:.2f}_big.p")


if __name__ == "__main__":
    if not demo_mode:
        run_dgh(5.5, k=1, n_periods=20)

    else:
        # For demo purposes, load the results from a previously-completed run
        m = load_data(
            os.path.join(
                "saved_data", "pic1", "2021-05-23_40920.687154_dgh_5.50_big.p"
            )
        )
        plots.plot_initial_distribution(m)
        plots.animate_phase_space(
            m,
            plot_title=(
                "Dory-Guest-Harris Instability: $k v_0 / \omega_c = 5.5$"
            ),
            repeat=True,
            hold=True,
        )
