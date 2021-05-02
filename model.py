"""
Coding project 1: Electrostatic Particle in Cell.
"""
import math

import numba
import numpy as np
import progressbar

from configuration import Configuration, ParticleData
from weighting import weight_particles, weight_field
from poisson import compute_field


@numba.njit(nogil=True)
def rotate_xy(x, y, theta):
    """Rotate 1D arrays x, y by angle theta. Do it fast."""
    ct = math.cos(theta)
    st = math.sin(theta)
    x2 = ct * x - st * y
    y2 = st * x + ct * y
    # for i in numba.prange(x.shape[0]):
    # y[i] = y2[i]
    # x[i] = x2[i]
    return (x2, y2)


class PicModel:
    """PIC model. The vast majority of the pic1 functionality lives here."""

    has_run = False

    def __init__(self, c: Configuration):
        self.c = c
        d = ParticleData(c)

        # Normalize particle positions to [0, 1]
        d.x_i = (c.initial_x - c.x_min) / c.L
        d.vx_i = c.initial_vx / c.L
        d.vy_i = c.initial_vy / c.L
        d.initial_x_norm = d.x_i
        d.initial_vx_norm = d.vx_i
        d.initial_vy_norm = d.vy_i

        # Initialize v_x, v_y with a 1/2 rotation backwards
        (d.vx_i, d.vy_i) = rotate_xy(d.vx_i, d.vy_i, c.wc * c.dt / 2)

        # Initialize v_x with a 1/2 step backwards
        rho = (
            c.eps0
            * weight_particles(
                d.x_i, c.x_j, c.dx, c.M, c.q, order=c.weighting_order
            )
            + c.rho_bg
        )
        d.initial_rho = rho
        # Solve for field at t=0
        d.ex_j = compute_field(rho, c.inv_a, c.dx)
        d.initial_ex = d.ex_j
        # Weight field to grid and accelerate particles
        ex_i = weight_field(d.x_i, c.x_j, d.ex_j, c.dx, order=c.weighting_order)
        d.vx_i -= (c.dt / 2) * c.qm * ex_i

        self.d = d

    # Needs to be a static method since @numba.njit won't work for class methods
    @staticmethod
    @numba.njit(nogil=True)
    def time_step(
        frame,
        x_i,
        vx_i,
        vy_i,
        x_j,
        ke_hist,
        fe_hist,
        p_hist,
        x_hist,
        vx_hist,
        vy_hist,
        ex_hist,
        ey_hist,
        subsample_ratio,
        M,
        dx,
        dt,
        eps0,
        rho_bg,
        q,
        qm,
        m,
        wc,
        weighting_order,
        inv_a,
        nonumba=False,
    ):
        """Evolve state forward in time by ∆t with periodic boundary conditions.

        Governing equations are:
            x[i](t + ∆t) = x[i](t) + v[i](t + ∆t/2) * ∆t
            v[i](t + ∆t) = v[i](t) + e[i](t + ∆t/2) * (q/m) * ∆t

        The @numba.jit decorator compiles the particle_push function to
        optimized machine code using the `llvmlite` version of the LLVM
        compiler. Depending on the size of n and t_steps, the performance
        improvement is up to 100x the speed of the pure Python (still awful
        compared to C, but good enough for educational purposes).

        The steps are:
        1.  Weight particle positions to the grid.
        2.  Solve for fields at the grid points. Compute field energy.
        3.  Weight fields to particles.
        4.  Half-accelerate v_x. Compute kinetic energy, momentum.
        4.1 Strang splitting. Rotate v_y, v_y.
        5.  Half-accelerate v_x.
        6.  Push position.
        """
        rho = (
            eps0 * weight_particles(x_i, x_j, dx, M, q, order=weighting_order)
            + rho_bg
        )

        # Solve Poisson's equation
        ex_j = compute_field(rho, inv_a, dx)
        for j in range(M):
            ex_hist[j, int(frame / subsample_ratio)] = ex_j[j]

        # Calculate total electric field energy
        fe_hist[frame] += dx / 2 * eps0 * np.sum(ex_j * ex_j)

        # Weight field on grid to particles
        e_i = weight_field(x_i, x_j, ex_j, dx, order=weighting_order)

        # Calculate what acceleration will be
        dv = dt * qm * e_i

        # Accelerate and push
        # x[i] and v[i] are offset in time by ∆t/2, so that they leap-frog past
        # each other:
        #
        #          x(old)       x(new)
        # -----------*------------*----->
        #   v(old)       v(new)
        # ----*------------*------------> t
        #     |      |     |      |
        #   -∆t/2    0   ∆t/2    ∆t

        # Half acceleration, then rotation by -wc*dt
        (vx_ip, vy_ip) = rotate_xy(vx_i + dv / 2, vy_i, -wc * dt)

        # Compute kinetic energy, momentum
        # KE = 0.5 * m * (v_x(t-1/2)*v_x(t+1/2) + v_y(t-1/2)*v_y(t+1/2))
        ke_hist[frame] += (
            m / 2 * np.sum(np.abs(vx_i * vx_ip) + np.abs(vy_i * vy_ip))
        )
        p_hist[frame] += m * np.sum((vx_i + vx_ip) / 2)
        vx_i += vx_ip - vx_i
        vy_i += vy_ip - vy_i

        # Second half acceleration

        # Push particles forward, now that we have v_i[n+1/2]

        for i in numba.prange(x_i.size):
            # numba.prange is like np.arange, but optimized for parallelization
            # across multiple CPU cores

            x_i[i] += vx_i[i] * dt
            # Apply periodic boundary conditions. NumPy uses the definition of
            # floor where floor(-2.5) == -3.
            x_i[i] -= np.floor(x_i[i])
            if frame % subsample_ratio == 0:
                sampled_frame = int(frame / subsample_ratio)
                x_hist[i][sampled_frame] += x_i[i]
                vx_hist[i][sampled_frame] += vx_i[i]
                vy_hist[i][sampled_frame] += vy_i[i]

    def run(self, showprogress=True):
        """Run the simulation."""
        c = self.c
        d = self.d
        if showprogress:
            print("Simulating...")
            bar = progressbar.ProgressBar(
                maxval=c.t_steps,
                widgets=[
                    progressbar.Bar("=", "[", "]"),
                    " ",
                    progressbar.Percentage(),
                ],
            )
            bar.start()
        for frame in range(c.t_steps):
            self.time_step(
                frame,
                x_i=d.x_i,
                vx_i=d.vx_i,
                vy_i=d.vy_i,
                x_j=c.x_j,
                ke_hist=d.ke_hist,
                fe_hist=d.fe_hist,
                p_hist=d.p_hist,
                x_hist=d.x_hist,
                vx_hist=d.vx_hist,
                vy_hist=d.vy_hist,
                ex_hist=d.ex_hist,
                ey_hist=d.ey_hist,
                subsample_ratio=c.subsample_ratio,
                M=c.M,
                dx=c.dx,
                dt=c.dt,
                eps0=c.eps0,
                rho_bg=c.rho_bg,
                q=c.q,
                qm=c.qm,
                m=c.m,
                wc=c.wc,
                weighting_order=c.weighting_order,
                inv_a=c.inv_a,
            )
            if showprogress:
                bar.update(frame)
        if showprogress:
            bar.finish()
            print("done!")
        self.has_run = True
