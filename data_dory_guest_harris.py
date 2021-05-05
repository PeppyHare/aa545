"Calculate frequencies from completed calls to run_dory_guest_harris."

import os
import pickle

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from util import load_data, count_crossings, save_plot
from model import PicModel
from configuration import Configuration, ParticleData
from run_dory_guest_harris import DGHConfiguration
import plots


def analyze_dgh(save_file, param):
    fn = os.path.join("saved_data", "dgh", save_file)
    m = load_data(fn)
    c = m.c
    d = m.d
    print(
        f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
        f" {m.has_run}"
    )
    ax_energy = plots.plot_energy_history(
        m, plot_title=f"$k v_0 / \omega_c = {param}$", hold=False
    )
    t_steps = c.t_steps
    # ke_hist = d.ke_hist
    fe_hist = d.fe_hist
    time_axis = c.time_axis

    # Let's try to identify the linear region of exponential growth
    # First, how many decades do we grow over? Compare the first 5% to the last
    slice_width = int(t_steps / 20)
    min_avg = np.max(fe_hist[0:slice_width])
    print(f"Initial slice max: {min_avg}")
    max_avg = np.average(fe_hist[-slice_width:])
    print(f"Final slice min: {max_avg}")
    llim = min_avg * 10
    ulim = max_avg / 2
    t_start_idx = 0
    t_end_idx = t_steps - 1
    for idx in range(t_steps):
        if fe_hist[idx] > llim:
            t_start_idx = idx
            break
    for idx in range(t_steps):
        if fe_hist[idx] > ulim:
            t_end_idx = idx
            break
    t_start = t_start_idx * c.dt
    t_end = t_end_idx * c.dt
    print(f"Identified linear growth between t={t_start:.2f} and t={t_end:.2f}")
    # t_start = 57.2
    # t_start_idx = int(t_start / c.dt) - 1
    # t_end = 183
    # t_end_idx = int(t_end // c.dt) - 1
    time_axis = c.time_axis[t_start_idx:t_end_idx]
    fe_range = m.d.fe_hist[t_start_idx:t_end_idx]

    lr = stats.linregress(time_axis, np.log(fe_range))
    ax_energy.plot(
        c.time_axis,
        np.exp(lr.slope * c.time_axis + lr.intercept),
        "r--",
        linewidth=0.5,
        label="fit",
    )
    print(f"Im(w/wc): {lr.slope / 2 / c.wc}")
    e_scaled = fe_range / np.exp(lr.slope * time_axis + lr.intercept)
    plt.figure()
    e_fft = np.real(np.fft.rfft(e_scaled))
    k_vec = np.fft.rfftfreq(len(e_scaled))
    plt.plot(k_vec[:100], e_fft[:100], ".", label="Re(w)")
    topk = np.argpartition(e_fft, -4)[-4:]
    for idx in topk:
        print(
            f"k={k_vec[idx]}, fft[k]={e_fft[idx]},"
            f" w={2 * np.pi * k_vec[idx] / c.dt}"
        )
    w_re = (
        count_crossings(e_scaled)
        / 4
        / (c.n_periods * (t_end - t_start) / (c.t_max))
    )
    print(f"Re(w/wc): {w_re / c.wc}")

    save_plot(f"dgh_{param}.pdf")
    plt.show()


if __name__ == "__main__":
    # analyze_dgh("4.10.p", 4.1)
    analyze_dgh("6.0-2.p", 6.0)


# if "4.1" in runs:
#     print("Normalized wave number: 4.10")
#     fn = os.path.join("saved_data", "dgh", "4.10.p")
#     m = load_data(fn)
#     c = m.c
#     print(
#         f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
#         f" {m.has_run}"
#     )
#     ax_energy = plots.plot_energy_history(
#         m, plot_title=f"$k v_0 / \omega_c = 4.1$", hold=False
#     )
#     t_start = 57.2
#     t_start_idx = int(t_start / c.dt) - 1
#     t_end = 183
#     t_end_idx = int(t_end // c.dt) - 1
#     time_axis = c.time_axis[t_start_idx:t_end_idx]
#     fe_range = m.d.fe_hist[t_start_idx:t_end_idx]

#     lr = stats.linregress(time_axis, np.log(fe_range))
#     ax_energy.plot(
#         c.time_axis,
#         np.exp(lr.slope * c.time_axis + lr.intercept),
#         "r--",
#         linewidth=0.5,
#         label="fit",
#     )
#     print(f"Im(w/wc): {lr.slope / 2 / c.wc}")
#     e_scaled = fe_range / np.exp(lr.slope * time_axis + lr.intercept)
#     w_re = (
#         count_crossings(e_scaled)
#         / 4
#         / (c.n_periods * (t_end - t_start) / (c.t_max))
#     )
#     print(f"Re(w/wc): {w_re / c.wc}")

#     save_plot("dgh_4.10.pdf")
#     plt.show()

# elif "4.5" in runs:
#     print("Normalized wave number: 4.5")
#     fn = os.path.join("saved_data", "dgh", "4.5.p")
#     m = load_data(fn)
#     c = m.c
#     print(
#         f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
#         f" {m.has_run}"
#     )
#     ax_energy = plots.plot_energy_history(
#         m, plot_title=f"$k v_0 / \omega_c = 4.5$", hold=False
#     )
#     t_start = 37.6
#     t_start_idx = int(t_start / c.dt) - 1
#     t_end = 131
#     t_end_idx = int(t_end // c.dt) - 1
#     time_axis = c.time_axis[t_start_idx:t_end_idx]
#     fe_range = m.d.fe_hist[t_start_idx:t_end_idx]

#     lr = stats.linregress(time_axis, np.log(fe_range))
#     ax_energy.plot(
#         c.time_axis,
#         np.exp(lr.slope * c.time_axis + lr.intercept),
#         "r--",
#         linewidth=0.5,
#         label="fit",
#     )
#     print(f"Im(w/wc): {lr.slope / 2 / c.wc}")

#     e_scaled = fe_range / np.exp(lr.slope * time_axis + lr.intercept)
#     w_re = (
#         count_crossings(e_scaled)
#         / 4
#         / (c.n_periods * (t_end - t_start) / (c.t_max))
#     )
#     print(f"Re(w/wc): {w_re / c.wc}")
#     plt.show()

# elif "5.6" in runs:
#     print("Normalized wave number: 5.6")
#     fn = os.path.join("saved_data", "dgh", "5.6.p")
#     m = load_data(fn)
#     c = m.c
#     print(
#         f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
#         f" {m.has_run}"
#     )
#     ax_energy = plots.plot_energy_history(
#         m, plot_title=f"$k v_0 / \omega_c = 5.6$", hold=False
#     )
#     t_start = 34
#     t_start_idx = int(t_start / c.dt) - 1
#     t_end = 113
#     t_end_idx = int(t_end // c.dt) - 1
#     time_axis = c.time_axis[t_start_idx:t_end_idx]
#     fe_range = m.d.fe_hist[t_start_idx:t_end_idx]

#     lr = stats.linregress(time_axis, np.log(fe_range))
#     ax_energy.plot(
#         c.time_axis,
#         np.exp(lr.slope * c.time_axis + lr.intercept),
#         "r--",
#         linewidth=0.5,
#         label="fit",
#     )
#     print(f"Im(w/wc): {lr.slope / 2 / c.wc}")

#     e_scaled = fe_range / np.exp(lr.slope * time_axis + lr.intercept)
#     w_re = (
#         count_crossings(e_scaled)
#         / 4
#         / (c.n_periods * (t_end - t_start) / (c.t_max))
#     )
#     print(f"Re(w/wc): {w_re / c.wc}")
#     plt.show()

# elif "6.0" in runs:
#     print("Normalized wave number: 6.0")
#     fn = os.path.join("saved_data", "dgh", "6.0.p")
#     m = load_data(fn)
#     c = m.c
#     print(
#         f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
#         f" {m.has_run}"
#     )
#     ax_energy = plots.plot_energy_history(
#         m, plot_title=f"$k v_0 / \omega_c = 6.0$", hold=False
#     )
#     t_start = 45
#     t_start_idx = int(t_start / c.dt) - 1
#     t_end = 179
#     t_end_idx = int(t_end // c.dt) - 1
#     time_axis = c.time_axis[t_start_idx:t_end_idx]
#     fe_range = m.d.fe_hist[t_start_idx:t_end_idx]

#     lr = stats.linregress(time_axis, np.log(fe_range))
#     ax_energy.plot(
#         c.time_axis,
#         np.exp(lr.slope * c.time_axis + lr.intercept),
#         "r--",
#         linewidth=0.5,
#         label="fit",
#     )
#     print(f"Im(w/wc): {lr.slope / 2 / c.wc}")

#     e_scaled = fe_range / np.exp(lr.slope * time_axis + lr.intercept)
#     w_re = (
#         count_crossings(e_scaled)
#         / 4
#         / (c.n_periods * (t_end - t_start) / (c.t_max))
#     )
#     print(f"Re(w/wc): {w_re / c.wc}")
#     plt.show()
