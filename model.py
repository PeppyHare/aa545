"""
Coding project 1: Electrostatic Particle in Cell.
"""
# from matplotlib.animation import FuncAnimation
import numba
import numpy as np
import progressbar

# from scipy import stats
from matplotlib import pyplot as plt

from configuration import Configuration, ParticleData
from weighting import weight_particles, weight_field
from poisson import compute_field


class PicModel:
    """PIC model. The vast majority of the pic1 functionality lives here."""

    has_run = False

    def __init__(self, c: Configuration):
        self.c = c
        d = ParticleData(c)

        # Normalize particle positions to [0, 1]
        d.x_i = (c.initial_x - c.x_min) / c.L
        d.v_i = c.initial_v / c.L

        # Initialize leapfrog with a 1/2 step backwards
        rho = (
            c.eps0
            * weight_particles(
                d.x_i, c.x_j, c.dx, c.M, c.q, order=c.weighting_order
            )
            + c.rho_bg
        )
        # Solve for field at t=0
        d.e_j = compute_field(rho, c.inv_a, c.dx)
        # Weight field to grid and accelerate particles
        e_i = weight_field(d.x_i, c.x_j, d.e_j, c.dx, order=c.weighting_order)
        d.v_i -= (c.dt / 2) * c.qm * e_i

        self.d = d

    # This needs to be a static method since @numba.njit won't work for
    # class methods
    @staticmethod
    @numba.njit
    def time_step(
        frame,
        x_i,
        v_i,
        x_j,
        ke_hist,
        fe_hist,
        p_hist,
        x_hist,
        v_hist,
        efield_hist,
        subsample_ratio,
        M,
        dx,
        dt,
        eps0,
        rho_bg,
        q,
        qm,
        m,
        weighting_order,
        inv_a,
        nonumba=False,
    ):
        """Evolve state forward in time by ∆t with periodic boundary conditions.

        Governing equations are:
            x[i](t + ∆t) = x[i](t) + v[i](t + ∆t/2) * ∆t
            v[i](t + ∆t) = v[i](t) + e[i](t + ∆t/2) * (q/m) * ∆t

        The @numba.jit decorator compiles the particle_push function to optimized
        machine code using the `llvmlite` version of the LLVM compiler. Depending on
        the size of n and t_steps, the performance improvement is up to 100x the
        speed of the pure Python (still awful compared to C, but good enough for
        educational purposes).

        The steps are:
        1. Weight particle positions to the grid.
        2. Solve for fields at the grid points. Compute field energy.
        3. Weight fields to particles.
        4. Half-accelerate velocity. Compute kinetic energy, momentum.
        5. Half-accelerate velocity.
        6. Push position.
        """
        rho = (
            eps0 * weight_particles(x_i, x_j, dx, M, q, order=weighting_order)
            + rho_bg
        )

        # Solve Poisson's equation
        e_j = compute_field(rho, inv_a, dx)
        for j in range(M):
            efield_hist[j, int(frame / subsample_ratio)] = e_j[j]

        # Calculate total electric field energy
        fe_hist[frame] += dx / 2 * eps0 * np.sum(e_j * e_j)

        # Weight field on grid to particles
        e_i = weight_field(x_i, x_j, e_j, dx, order=weighting_order)

        # Calculate what acceleration will be
        dv = dt * qm * e_i

        # Compute kinetic energy, momentum
        ke_hist[frame] += m / 2 * np.sum(np.abs(v_i * (v_i + dv)))
        p_hist[frame] += m * np.sum(v_i + (dv / 2))

        # Accelerate and push
        # x[i] and v[i] are offset in time by ∆t/2, so that they leap-frog past each
        # other:
        #
        #          x(old)       x(new)
        # -----------*------------*----->
        #   v(old)       v(new)
        # ----*------------*------------> t
        #     |      |     |      |
        #   -∆t/2    0   ∆t/2    ∆t

        # Accelerate
        v_i += dv

        # Push particles forward, now that we have v_i[n+1/2]
        # print(frame, subsample_ratio, int(frame/subsample_ratio) - 1, ke_hist.size, x_hist.shape)
        for i in numba.prange(x_i.size):
            # numba.prange is like np.arange, but optimized for parallelization
            # across multiple CPU cores

            x_i[i] += v_i[i] * dt
            # Apply periodic boundary conditions. NumPy uses the definition of floor
            # where floor(-2.5) == -3.
            x_i[i] -= np.floor(x_i[i])
            if frame % subsample_ratio == 0:
                x_hist[i][int(frame / subsample_ratio)] += x_i[i]
                v_hist[i][int(frame / subsample_ratio)] += v_i[i]

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
                v_i=d.v_i,
                x_j=c.x_j,
                ke_hist=d.ke_hist,
                fe_hist=d.fe_hist,
                p_hist=d.p_hist,
                x_hist=d.x_hist,
                v_hist=d.v_hist,
                efield_hist=d.efield_hist,
                subsample_ratio=c.subsample_ratio,
                M=c.M,
                dx=c.dx,
                dt=c.dt,
                eps0=c.eps0,
                rho_bg=c.rho_bg,
                q=c.q,
                qm=c.qm,
                m=c.m,
                weighting_order=c.weighting_order,
                inv_a=c.inv_a,
            )
            if showprogress:
                bar.update(frame)
        if showprogress:
            bar.finish()
            print("done!")
        self.has_run = True
