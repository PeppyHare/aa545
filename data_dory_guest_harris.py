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


runs = ["4.1"]

if "4.1" in runs:
    print("Normalized wave number: 4.10")
    fn = os.path.join("saved_data", "dgh", "4.10.p")
    m = load_data(fn)
    c = m.c
    print(
        f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
        f" {m.has_run}"
    )
    ax_energy = plots.plot_energy_history(
        m, plot_title=f"$k v_0 / \omega_c = 4.1$", hold=False
    )
    t_start = 57.2
    t_start_idx = int(t_start / c.dt) - 1
    t_end = 183
    t_end_idx = int(t_end // c.dt) - 1
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
    w_re = (
        count_crossings(e_scaled)
        / 4
        / (c.n_periods * (t_end - t_start) / (c.t_max))
    )
    print(f"Re(w/wc): {w_re / c.wc}")

    save_plot("dgh_4.10.pdf")
    plt.show()

elif "4.5" in runs:
    print("Normalized wave number: 4.5")
    fn = os.path.join("saved_data", "dgh", "4.5.p")
    m = load_data(fn)
    c = m.c
    print(
        f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
        f" {m.has_run}"
    )
    ax_energy = plots.plot_energy_history(
        m, plot_title=f"$k v_0 / \omega_c = 4.5$", hold=False
    )
    t_start = 37.6
    t_start_idx = int(t_start / c.dt) - 1
    t_end = 131
    t_end_idx = int(t_end // c.dt) - 1
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
    w_re = (
        count_crossings(e_scaled)
        / 4
        / (c.n_periods * (t_end - t_start) / (c.t_max))
    )
    print(f"Re(w/wc): {w_re / c.wc}")
    plt.show()

elif "5.6" in runs:
    print("Normalized wave number: 5.6")
    fn = os.path.join("saved_data", "dgh", "5.6.p")
    m = load_data(fn)
    c = m.c
    print(
        f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
        f" {m.has_run}"
    )
    ax_energy = plots.plot_energy_history(
        m, plot_title=f"$k v_0 / \omega_c = 5.6$", hold=False
    )
    t_start = 34
    t_start_idx = int(t_start / c.dt) - 1
    t_end = 113
    t_end_idx = int(t_end // c.dt) - 1
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
    w_re = (
        count_crossings(e_scaled)
        / 4
        / (c.n_periods * (t_end - t_start) / (c.t_max))
    )
    print(f"Re(w/wc): {w_re / c.wc}")
    plt.show()

elif "6.0" in runs:
    print("Normalized wave number: 6.0")
    fn = os.path.join("saved_data", "dgh", "6.0.p")
    m = load_data(fn)
    c = m.c
    print(
        f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
        f" {m.has_run}"
    )
    ax_energy = plots.plot_energy_history(
        m, plot_title=f"$k v_0 / \omega_c = 6.0$", hold=False
    )
    t_start = 45
    t_start_idx = int(t_start / c.dt) - 1
    t_end = 179
    t_end_idx = int(t_end // c.dt) - 1
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
    w_re = (
        count_crossings(e_scaled)
        / 4
        / (c.n_periods * (t_end - t_start) / (c.t_max))
    )
    print(f"Re(w/wc): {w_re / c.wc}")
    plt.show()
