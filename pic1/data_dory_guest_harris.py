"Calculate frequencies from completed calls to run_dory_guest_harris."

import os

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import scipy.special as sp
import matplotlib.image as mpimg

from pic1.util import load_data, count_crossings, save_plot
import pic1.plots as plots

# These classes are not used directly, but must be imported in order to unpack
# the pickled data from previous runs
from pic1.model import PicModel
from pic1.configuration import Configuration, ParticleData
from pic1.run_dory_guest_harris import DGHConfiguration


demo_mode = True


def analyze_dgh(save_file, param):
    fn = os.path.join("saved_data", "dgh", save_file)
    m = load_data(fn)
    c = m.c
    d = m.d
    print(
        f"N: {c.N}, M: {c.M}, wp: {c.wp:.3f}, wc: {c.wc:.3f}, has_run:"
        f" {m.has_run}"
    )
    if not demo_mode:
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
    max_avg = np.average(fe_hist[-slice_width:])
    llim = min_avg * 10
    ulim = max_avg / 2

    # Step over the growth of the field energy. If we go above 10x the max field
    # energy in the first 5%, then we are in the linear growth phase. When we
    # reach half of the saturation value, we leave the linear growth phase.
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

    time_axis = c.time_axis[t_start_idx:t_end_idx]
    fe_range = m.d.fe_hist[t_start_idx:t_end_idx]
    if fe_range.size > 0:
        print(
            f"Identified linear growth between t={t_start:.2f} and"
            f" t={t_end:.2f}"
        )
        lr = stats.linregress(time_axis, np.log(fe_range))
        if not demo_mode:
            ax_energy.plot(
                c.time_axis,
                np.exp(lr.slope * c.time_axis + lr.intercept),
                "r--",
                linewidth=0.5,
                label="fit",
            )
        print(f"Im(w/wc): {lr.slope / 2 / c.wc}")
        imw = lr.slope / 2 / c.wc
        e_scaled = fe_range / np.exp(lr.slope * time_axis + lr.intercept)
        e_fft = np.real(np.fft.rfft(e_scaled))
        k_vec = np.fft.rfftfreq(len(e_scaled))
        topk = np.argpartition(e_fft, -4)[-4:]
        if not demo_mode:
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
        rew = w_re / c.wc
        if not demo_mode:
            plots.plot_snapshots_velocity_phase_space(
                m,
                hold=False,
                plot_title=(
                    f"Velocity snapshots ($k v_\perp / \omega_c = {param},"
                    " \omega_p ^2 /\omega_c ^2 = 10$)"
                ),
            )
            save_plot(f"dgh_{param}.pdf")
            plt.show()
    else:
        print("No exponential growth detected.")
        imw = 0
        rew = 0
        e_fft = np.fft.rfft(d.fe_hist)
        k_vec = np.fft.rfftfreq(len(d.fe_hist))
        dw = 2 * np.pi / c.t_max
        n_t = d.fe_hist.size
        mode = k_vec[np.argmax(e_fft[1:])]
        print(f"Re(w/wc): {mode * n_t * dw}")
        rew = mode * n_t * dw
        if not demo_mode:
            topk = np.argpartition(e_fft, -10)[-10:]
            for idx in topk:
                print(
                    f"k={k_vec[idx]}, fft[k]={e_fft[idx]},"
                    f" w={k_vec[idx] * n_t * dw}"
                )
            plt.figure()
            plt.plot(
                k_vec[:100] * n_t * dw, e_fft[:100] / c.wc, "-", label="Re(w)"
            )
            plt.xlabel(r"$\omega / \omega_c$")
            plt.ylabel(r"$FFT(U_E)$")
            plt.show()
    return (imw, rew)


def compare_grid_spacing(files, labels):
    fig = plt.figure(figsize=(12, 12))
    ax_energy = fig.add_subplot(2, 1, (1, 2))
    ax_energy.set_title(r"DGH Instability - Grid Spacing Comparison")
    ax_energy.set_ylabel(r"$U_E$")
    ax_energy.set_xlabel("time")
    ax_energy.set_yscale("log")
    for idx in range(len(files)):
        fn = os.path.join("saved_data", "dgh", files[idx])
        m = load_data(fn)
        c = m.c
        d = m.d
        time_axis = c.time_axis
        fe_hist = d.fe_hist
        ax_energy.plot(time_axis, fe_hist, "-", markersize=1, label=labels[idx])
    ax_energy.legend(loc="lower right")
    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    save_plot("dgh_grid_spacing_comparison.pdf")
    plt.show()


if __name__ == "__main__":
    # The filenames passed in refer to previous data runs generated by
    # run_dgh.py and  stored in ./saved_data/dgh
    imw = []
    rew = []
    trials = [4.1, 4.5, 5.0, 5.6, 6.0, 6.6]
    im, re = analyze_dgh("4.10.p", 4.1)
    imw.append(im)
    rew.append(re)
    im, re = analyze_dgh("4.50_16_32.p", 4.5)
    imw.append(im)
    rew.append(re)
    im, re = analyze_dgh("5.0-5.p", 5.0)
    imw.append(im)
    rew.append(re)
    im, re = analyze_dgh("5.6.p", 5.6)
    imw.append(im)
    rew.append(re)
    im, re = analyze_dgh("6.0.p", 6.0)
    imw.append(im)
    rew.append(re)
    im, re = analyze_dgh("6.6.p", 6.6)
    imw.append(im)
    rew.append(re)

    # Calculate exact real part of dispersion relation. Imaginary part proves
    # very hard to do...

    # quad_arr = np.array(
    #     [
    #         [1, -0.9988664044200710501855, 0.002908622553155140958],
    #         [2, -0.994031969432090712585, 0.0067597991957454015028],
    #         [3, -0.985354084048005882309, 0.0105905483836509692636],
    #         [4, -0.9728643851066920737133, 0.0143808227614855744194],
    #         [5, -0.9566109552428079429978, 0.0181155607134893903513],
    #         [6, -0.9366566189448779337809, 0.0217802431701247929816],
    #         [7, -0.9130785566557918930897, 0.02536067357001239044],
    #         [8, -0.8859679795236130486375, 0.0288429935805351980299],
    #         [9, -0.8554297694299460846114, 0.0322137282235780166482],
    #         [10, -0.821582070859335948356, 0.0354598356151461541607],
    #         [11, -0.784555832900399263905, 0.0385687566125876752448],
    #         [12, -0.744494302226068538261, 0.041528463090147697422],
    #         [13, -0.70155246870682225109, 0.044327504338803275492],
    #         [14, -0.6558964656854393607816, 0.0469550513039484329656],
    #         [15, -0.6077029271849502391804, 0.0494009384494663149212],
    #         [16, -0.5571583045146500543155, 0.0516557030695811384899],
    #         [17, -0.5044581449074642016515, 0.0537106218889962465235],
    #         [18, -0.449806334974038789147, 0.05555774480621251762357],
    #         [19, -0.3934143118975651273942, 0.057189925647728383723],
    #         [20, -0.335500245419437356837, 0.058600849813222445835],
    #         [21, -0.2762881937795319903276, 0.05978505870426545751],
    #         [22, -0.2160072368760417568473, 0.0607379708417702160318],
    #         [23, -0.1548905899981459020716, 0.06145589959031666375641],
    #         [24, -0.0931747015600861408545, 0.0619360674206832433841],
    #         [25, -0.0310983383271888761123, 0.062176616655347262321],
    #         [26, 0.0310983383271888761123, 0.062176616655347262321],
    #         [27, 0.09317470156008614085445, 0.0619360674206832433841],
    #         [28, 0.154890589998145902072, 0.0614558995903166637564],
    #         [29, 0.2160072368760417568473, 0.0607379708417702160318],
    #         [30, 0.2762881937795319903276, 0.05978505870426545751],
    #         [31, 0.335500245419437356837, 0.058600849813222445835],
    #         [32, 0.3934143118975651273942, 0.057189925647728383723],
    #         [33, 0.4498063349740387891471, 0.055557744806212517624],
    #         [34, 0.5044581449074642016515, 0.0537106218889962465235],
    #         [35, 0.5571583045146500543155, 0.05165570306958113849],
    #         [36, 0.60770292718495023918, 0.049400938449466314921],
    #         [37, 0.6558964656854393607816, 0.046955051303948432966],
    #         [38, 0.7015524687068222510896, 0.044327504338803275492],
    #         [39, 0.7444943022260685382605, 0.0415284630901476974224],
    #         [40, 0.7845558329003992639053, 0.0385687566125876752448],
    #         [41, 0.8215820708593359483563, 0.0354598356151461541607],
    #         [42, 0.8554297694299460846114, 0.0322137282235780166482],
    #         [43, 0.8859679795236130486375, 0.02884299358053519803],
    #         [44, 0.9130785566557918930897, 0.02536067357001239044],
    #         [45, 0.9366566189448779337809, 0.0217802431701247929816],
    #         [46, 0.9566109552428079429978, 0.0181155607134893903513],
    #         [47, 0.9728643851066920737133, 0.0143808227614855744194],
    #         [48, 0.985354084048005882309, 0.010590548383650969264],
    #         [49, 0.9940319694320907125851, 0.0067597991957454015028],
    #         [50, 0.9988664044200710501855, 0.0029086225531551409584],
    #     ]
    # )

    # # Define integrand
    # def integrand(x, om, mu):
    #     # affine transformation to prepare quad method [-1,1] -> [0, pi]
    #     theta = 0.5 * np.pi * (1.0 + x)
    #     # build integrand, z = 2 * mu * cos(theta), shape (quad points, k, real om, imag om)
    #     z = np.tensordot(
    #         2.0 * np.tensordot(np.cos(0.5 * theta), mu, axes=0),
    #         np.ones_like(om),
    #         axes=0,
    #     )
    #     # compute J0(z)
    #     J0 = sp.jv(0, z)
    #     # compute product theta * frequency
    #     t_f = np.tensordot(theta, om, axes=0)
    #     # compute product sin(theta) * sin(theta * frequency)
    #     sine_t_f = np.multiply(np.sin(theta)[:, None, None], np.sin(t_f))
    #     # return the integrand evaluated on the quadrature points
    #     return np.multiply(sine_t_f[:, None, :, :], J0)

    # # make grids for plotting
    # k = np.linspace(0, 7.0, num=75)  # np.linspace(6, 7.0, num=75)
    # om_r = np.linspace(-1.0e-3, 5.0, num=75)  # np.linspace(3.2, 3.4, num=75)
    # om_i = np.linspace(0.0, 0.35, num=75)  # np.linspace(0.19, 0.2, num=75) #
    # om = np.tensordot(om_r, np.ones_like(om_i), axes=0) + 1.0j * np.tensordot(
    #     np.ones_like(om_r), om_i, axes=0
    # )

    # # evaluate function
    # a = np.sqrt(10)  # plasma-cyclotron frequency ratio
    # int_on_quads = integrand(x=quad_arr[:, 1], om=om, mu=k)
    # # do quadrature integral as inner product
    # f1 = 0.5 * np.pi  # scaling for quadrature integration [0, pi] -> [-1, 1]
    # result = f1 * np.tensordot(quad_arr[:, 2], int_on_quads, axes=([0], [0]))
    # # evaluate dispersion function, multiply by singular terms for better behaved function
    # D = np.sin(np.pi * om[None, :, :]) + (a ** 2.0) * result
    # print(D.shape)

    # # 2-grids for plotting
    # K = np.tensordot(k, np.ones_like(om_r), axes=0)
    # F = np.tensordot(np.ones_like(k), om_r, axes=0)

    # OR = np.tensordot(om_r, np.ones_like(om_i), axes=0)
    # OI = np.tensordot(np.ones_like(om_r), om_i, axes=0)

    plt.close("all")
    fig = plt.figure(figsize=(8, 8))
    dispersion_data = load_data(
        os.path.join(
            "saved_data", "dgh", "dory-guest-harris-dispersion-relation.p"
        )
    )
    D = dispersion_data["D"]
    K = dispersion_data["K"]
    F = dispersion_data["F"]
    plt.contour(
        K,
        F,
        np.real(D[:, :, 0]),
        0,
        linewidths=3,
    )
    plt.grid(True)
    plt.xlabel(r"$k v_0 / \omega_c$")
    plt.ylabel(r"$\omega / \omega_c$")
    # fig2 = plt.figure(figsize=(8, 8))
    plt.suptitle("DGH Growth Rates")
    # axes = fig2.add_subplot(111)
    plt.plot(trials, rew, "ro", label="Real (simulated)")
    plt.plot(trials, imw, "go", label="Imaginary (simulated)")
    plt.xlabel(r"$k v_0 / \omega_c$")
    plt.ylabel(r"$\omega / \omega_c$")
    plt.legend()
    plt.tight_layout()
    fig2 = plt.figure(figsize=(8, 8))
    img = mpimg.imread(os.path.join("saved_data", "dgh", "dispersion.png"))
    plt.imshow(img)
    plt.show()

    def grid3d(arr0, arr1, arr2):
        return np.tensordot(arr0, np.tensordot(arr1, arr2, axes=0), axes=0)

    compare_grid_spacing(
        ["4.50_16_32.p", "4.50_32_64.p", "4.50_64_128.p", "4.50_128_256.p"],
        ["16x32", "32x64", "64x128", "128x256"],
    )
