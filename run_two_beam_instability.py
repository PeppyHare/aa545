"""Two Stream Instability

Investigate the instability that arises when two electron beams counter stream.
The instability can be quantified by measuring the simulated plasma growth rate
and comparing it to that predicted by linear theory.
"""

import time

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from configuration import Configuration
from model import PicModel
import plots


class TwoStreamConfiguration(Configuration):
    N = 512
    plot_grid_lines = False
    n_periods = 20
    wp = 1

    def __init__(self, k, M):
        self.x_min = -np.pi / k
        self.x_max = np.pi / k
        self.M = M
        Configuration.__init__(self)

    def initialize_particles(self):
        v0 = 0.15
        dx = 0.01
        beam1_x = np.linspace(self.x_min, self.x_max, int(self.N / 2 + 1))[:-1]
        beam1_x += dx * np.sin(beam1_x)
        beam2_x = np.linspace(self.x_min, self.x_max, int(self.N / 2 + 1))[:-1]
        beam2_x -= dx * np.sin(beam2_x)
        beam1_v = v0 * np.ones_like(beam1_x)
        beam2_v = -v0 * np.ones_like(beam2_x)
        self.initial_x = np.concatenate([beam1_x, beam2_x])
        self.initial_v = np.concatenate([beam1_v, beam2_v])


def calc_growth_rate(k, M):
    """Growth rate of the two stream instability.

    Set up a PIC model for the given value k. Try to determine the exponential
    growth rate of the lowest unstable mode of the two-stream instability."""

    c = TwoStreamConfiguration(k, M)
    m = PicModel(c)
    # plot_initial_distribution(m)
    start = time.perf_counter()
    m.run()
    end = time.perf_counter()
    print(f"Total time: {(end-start)*1000:.4f}ms")

    v0 = 0.2
    wp = c.wp
    print(f"k: {k}, k*v0: {k*v0}, wp: {wp}")
    # Dispersion relation for opposing, equal-strength streams:
    # w = +/- [ k^2 v_0 ^2 + wp ^2 +/- wp*(4 k^2 v0^2 + wp ^2)**(1/2)]**(1/2)
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
    print(
        f"Possible w solutions:\n{sol1:.4f}\n{sol2:.4f}\n{sol3:.4f}\n{sol4:.4f}"
    )

    # Try to glean the growth rate of the lowest-order instability
    # Try and find where field energy goes above 10% of total energy
    d = m.d
    t_steps = c.t_steps
    ke_hist = d.ke_hist
    fe_hist = d.fe_hist
    time_axis = c.time_axis
    ramp_start = 0
    ramp_end = t_steps - 1
    for idx in range(t_steps):
        if fe_hist[idx] / ke_hist[idx] > 0.001:
            ramp_start = idx
            break
    for idx in range(t_steps):
        if fe_hist[idx] / ke_hist[idx] > 0.1:
            ramp_end = idx
            break
    dt = c.dt
    print(
        f"Max instability growth between steps {ramp_start} and {ramp_end} out"
        f" of {t_steps} (between t=[{ramp_start*dt}, {ramp_end*dt}])."
    )
    lr = stats.linregress(
        time_axis[ramp_start:ramp_end], np.log(fe_hist[ramp_start:ramp_end])
    )
    expected_rate = max([abs(sol1.imag), abs(sol2.imag)])
    print(
        "Expected growth rate: ",
        expected_rate,
        " measured growth rate: ",
        lr.slope,
    )
    plots.animate_phase_space(m, hold=False)
    plots.plot_traces(
        m, max_traces=40, start_at_frame=int(ramp_start / c.subsample_ratio)
    )
    # plot_snapshots(m, hold=False)
    # plot_energy_history(m, hold=False)
    # plt.show()
    return (expected_rate, lr.slope)


k_trials = [1, 2, 3, 4, 5, 6, 7, 8]
n_trials = len(k_trials)
expects = np.zeros(n_trials)
results = np.zeros(n_trials)
inputs = np.zeros(n_trials)

for idx, k in enumerate(k_trials):
    (expect, result) = calc_growth_rate(k, M=512)
    expects[idx] = expect
    results[idx] = result

fig = plt.figure()
plt.plot(k_trials, expects, "-o", color="cyan", label="expected")
plt.plot(k_trials, results, "o", color="orange", label="measured")
plt.legend()
plt.xlabel("k")
plt.ylabel(r"$\omega$")
plt.show()
