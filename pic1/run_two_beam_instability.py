"""Two Stream Instability

Investigate the instability that arises when two electron beams counter stream.
The instability can be quantified by measuring the simulated plasma growth rate
and comparing it to that predicted by linear theory.
"""

import time
import multiprocessing

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from configuration import Configuration
from model import PicModel
import plots
from util import save_plot
from weighting import weight_particles


class TwoStreamConfiguration(Configuration):
    plot_grid_lines = False
    wp = 1
    max_history_steps = 1000
    markersize = 4

    def __init__(
        self,
        k,
        M,
        N,
        perturbation=0.001,
        n_periods=12,
        dt=0.01,
        beam_velocity=0.2,
    ):
        self.M = M
        self.N = N
        self.k = k
        self.dt = dt
        self.x_min = -np.pi / k
        self.x_max = np.pi / k
        self.beam_velocity = beam_velocity
        self.perturbation = perturbation
        self.n_periods = n_periods
        self.weighting_order = 1
        Configuration.__init__(self)

    def set_initial_conditions(self):
        v0 = self.beam_velocity
        dx = self.perturbation
        beam1_x = np.linspace(self.x_min, self.x_max, int(self.N / 2) + 1)[:-1]
        shift = dx * np.sin(self.k * beam1_x)
        shift -= shift[0]
        beam1_x += shift
        beam2_x = np.linspace(self.x_min, self.x_max, int(self.N / 2) + 1)[:-1]
        # beam2_x -= shift
        beam1_v = v0 * np.ones_like(beam1_x)
        beam2_v = -v0 * np.ones_like(beam2_x)
        self.initial_x = np.concatenate([beam1_x, beam2_x])
        self.initial_vx = np.concatenate([beam1_v, beam2_v])
        self.initial_vy = np.zeros_like(self.initial_vx)


def calc_growth_rate(
    k,
    M=512,
    N=1024,
    ulim=10 ** -1,
    llim=10 ** -3,
    perturbation=0.001,
    # expects=[],
    # results=[],
    # k_trials=[],
    n_periods=24,
    dt=0.01,
    headless=True,
):
    """Growth rate of the two stream instability.

    Set up a PIC model for the given value k. Try to determine the exponential
    growth rate of the lowest unstable mode of the two-stream instability."""

    c = TwoStreamConfiguration(k, M, N, perturbation, n_periods, dt)
    m = PicModel(c)
    start = time.perf_counter()
    # m.run()
    end = time.perf_counter()
    print(f"Total time: {(end-start)*1000:.4f}ms")

    v0 = c.beam_velocity
    wp = c.wp
    print(f"k: {k}, k*v0/wp: {k*v0/wp:.2f}")
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
    # plots.plot_initial_distribution(m)
    # print(
    #     f"Possible w solutions:\n{sol1:.4f}\n{sol2:.4f}\n{sol3:.4f}\n{sol4:.4f}"
    # )

    # Plot the initial density displacement for each beam
    if not headless:
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"k={k}, k*v0/wp={k*v0/c.wp:.2f}, M={M}, N={c.N}")
        ax_init = fig.add_subplot(211)
        ax_init.set_title("Initial perturbation")
        ax_init.set_ylabel(r"$\rho$")
        ax_init.set_xlabel("x")
        halfway = int(c.N / 2)
        beam1_x = (c.initial_x[:halfway] - c.x_min) / c.L
        beam2_x = (c.initial_x[halfway:] - c.x_min) / c.L
        x_j = c.x_j
        x_j_unorm = (c.x_j * c.L) + c.x_min
        rho1 = (
            weight_particles(
                beam1_x, x_j, c.dx, c.M, q=c.q, order=c.weighting_order
            )
            - c.rho_bg / 2
        )
        rho2 = (
            weight_particles(
                beam2_x, x_j, c.dx, c.M, q=c.q, order=c.weighting_order
            )
            - c.rho_bg / 2
        )
        ax_init.plot(
            x_j_unorm,
            rho1 - np.average(rho1),
            ".",
            color="tab:orange",
            markersize=c.markersize,
            label="xv",
        )
        # ax_init.plot(
        #     x_j_unorm,
        #     rho2 - np.average(rho2),
        #     ".",
        #     color="tab:cyan",
        #     markersize=c.markersize,
        #     label="xv",
        # )
        ax_init.set_xlim(c.x_min, c.x_max)
        rmin = np.min([rho1 - np.average(rho1), rho2 - np.average(rho2)])
        rmax = np.max([rho1 - np.average(rho1), rho2 - np.average(rho2)])
        print(f"rmin: {rmin:.4f}, rmax: {rmax:.4f}")
        ax_init.set_ylim(rmin, rmax)
        # for grid_pt in x_j_unorm:
        #     ax_init.axvline(
        #         grid_pt,
        #         linestyle="--",
        #         color="k",
        #         linewidth=0.1,
        #     )
        ax_field = fig.add_subplot(212)
        ax_field.plot(m.d.initial_ex)
        save_plot(f"two_stream_initial_density_k={k}.pdf")
        plt.show()

    # Try to glean the growth rate of the lowest-order instability
    # Try and find where field energy goes above 10% of total energy
    m.run()
    d = m.d
    t_steps = c.t_steps
    ke_hist = d.ke_hist
    fe_hist = d.fe_hist
    time_axis = c.time_axis
    ramp_start = 0
    ramp_end = t_steps - 1
    for idx in range(t_steps):
        if fe_hist[idx] / ke_hist[idx] > llim:
            ramp_start = idx
            break
    for idx in range(t_steps):
        if fe_hist[idx] / ke_hist[idx] > ulim:
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
        lr.slope / 2,
    )

    ax_energy = plots.plot_traces(
        m,
        max_traces=25,
        start_at_frame=int(ramp_start / c.subsample_ratio),
        plot_title=f"Plot traces k={k}, k*v0/wp={k*v0/wp:.2f}",
        hold=False,
    )
    ax_energy.plot(
        time_axis,
        np.exp(lr.slope * time_axis + lr.intercept),
        "r--",
        linewidth=0.5,
        label="fit",
    )
    ax_energy.set_ylim(min(d.fe_hist), 10 * max(d.fe_hist + d.ke_hist))
    save_plot(f"two-stream-traces-k={k}.pdf")
    snapshots_title = (
        f"Snapshots: $k={k}, L={c.x_max - c.x_min:.2f},"
        f" kv_0/\omega_p={k*v0/wp:.2f}$"
    )
    plots.plot_snapshots(
        m,
        hold=False,
        plot_title=(snapshots_title),
        filename=f"snapshots_n={c.N}_k={k}.pdf",
    )
    if not headless:
        velocity_snapshots_title = (
            f"Evolution of $f(v)$: $k={k}, L={c.x_max - c.x_min:.2f},"
            f" kv_0/\omega_p={k*v0/wp:.2f}$"
        )
        plots.plot_snapshots_velocity(
            m,
            hold=True,
            plot_title=(velocity_snapshots_title),
        )
        energy_title = (
            f"Total Energy over Time: $k={k}, L={c.x_max - c.x_min:.2f},"
            f" kv_0/\omega_p={k*v0/wp:.2f}$"
        )
        ax_energy = plots.plot_energy_history(
            m, hold=False, plot_title=energy_title
        )

        plt.figure()
        plt.plot(time_axis, d.fe_hist / d.ke_hist)
        plt.yscale("log")

        plots.animate_phase_space(
            m,
            plot_title=(
                f"Phase space animation k: {k}, k*v0/wp: {k*v0/wp:.2f}, dx:"
                f" {c.perturbation}"
            ),
            repeat=True,
            hold=True,
        )
        plt.show()
    # expects.append(expected_rate)
    # results.append(lr.slope / 2)
    # k_trials.append(k)
    return (expected_rate, lr.slope / 2)


if __name__ == "__main__":

    calc_growth_rate(
        k=1,
        M=512,
        N=4096,
        ulim=10 ** -4,
        llim=10 ** -6,
        perturbation=0.0001,
        headless=False,
    )
    # calc_growth_rate(
    #     k=2,
    #     M=1024,
    #     N=1024,
    #     ulim=10 ** -1,
    #     llim=10 ** -3,
    #     perturbation=0.001,
    #     headless=False,
    # )
    # calc_growth_rate(
    #     k=3,
    #     M=1024,
    #     N=1024,
    #     ulim=10 ** -1,
    #     llim=10 ** -3,
    #     perturbation=0.0001,
    #     headless=False,
    #     n_periods=10,
    #     dt=0.01,
    # )
    # calc_growth_rate(
    #     k=4,
    #     M=2048,
    #     N=1024,
    #     ulim=10 ** -1,
    #     llim=10 ** -3,
    #     perturbation=0.0001,
    #     headless=False,
    #     n_periods=10,
    #     dt=0.01,
    # )
    # calc_growth_rate(
    #     k=5,
    #     M=2048,
    #     N=4096,
    #     ulim=10 ** -1,
    #     llim=10 ** -3,
    #     perturbation=0.0001,
    #     headless=False,
    #     n_periods=10,
    #     dt=0.01,
    # )
    # calc_growth_rate(
    #     k=6,
    #     M=2048,
    #     N=4096,
    #     ulim=10 ** -1,
    #     llim=10 ** -3,
    #     perturbation=0.0001,
    #     headless=False,
    #     n_periods=10,
    #     dt=0.01,
    # )
    # calc_growth_rate(
    #     k=7,
    #     M=4096,
    #     N=8192,
    #     ulim=10 ** -1,
    #     llim=10 ** -3,
    #     perturbation=0.0001,
    #     headless=False,
    #     n_periods=10,
    #     dt=0.01,
    # )
    # calc_growth_rate(
    #     k=8,
    #     M=4096,
    #     N=8192,
    #     ulim=10 ** -1,
    #     llim=10 ** -3,
    #     perturbation=0.0001,
    #     headless=False,
    #     n_periods=10,
    #     dt=0.01,
    # )

    k_trials = range(1, 8)

    with multiprocessing.Pool(
        min(len(k_trials), multiprocessing.cpu_count())
    ) as p:
        results = p.map(calc_growth_rate, k_trials)
        p.close()
    print(results)
    expected_w = []
    measured_w = []
    for result in results:
        expected_w.append(result[0])
        measured_w.append(result[1])

    fig = plt.figure()
    plt.plot(
        np.array(k_trials) * 0.2,
        expected_w,
        "-o",
        color="cyan",
        label="expected",
    )
    plt.plot(
        np.array(k_trials) * 0.2,
        measured_w,
        "o",
        color="orange",
        label="measured",
    )
    plt.title("Dispersion Comparison with Linear Theory")
    plt.legend()
    plt.xlabel(r"$kv_0/\omega_p$")
    plt.ylabel(r"$|\omega|$")
    save_plot("dispersion_two_stream.pdf")
    plt.show()
