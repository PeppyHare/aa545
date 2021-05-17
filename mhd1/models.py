"""Coding project 2: Ideal MHD."""
import numba
import numpy as np
import progressbar
import tables

from mhd1.configuration import Configuration, ParticleData


class MHDModel:
    """Attempt to solve a system under the equations of ideal MHD."""

    has_run = False

    def __init__(self, c: Configuration):
        """Initialize model."""
        self.c = c
        self.d = ParticleData(c)

    def write_out_data(self, Q):
        with tables.open_file(self.d.h5_filename, mode="a") as f:
            f.root.rho.append(
                Q[0, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )
            f.root.mx.append(
                Q[1, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )
            f.root.my.append(
                Q[2, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )
            f.root.mz.append(
                Q[3, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )
            f.root.bx.append(
                Q[4, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )
            f.root.by.append(
                Q[5, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )
            f.root.bz.append(
                Q[6, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )
            f.root.e.append(
                Q[7, :, :, :].reshape((1, self.c.Mx, self.c.My, self.c.Mz))
            )

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
        time_step_method = c.time_step_method
        for n in numba.prange(c.t_steps):
            time_step_method(
                Q=d.Q,
                dx=c.dx,
                dy=c.dy,
                dz=c.dz,
                dt=c.dt,
                gamma=c.gamma,
                mu=c.mu,
                bcx=c.bcx,
                bcy=c.bcy,
                bcz=c.bcz,
            )
            if n % c.subsample_ratio == 0 and n > 0:
                self.write_out_data(d.Q)

            if showprogress:
                bar.update(n)
        if showprogress:
            bar.finish()
        print("done!")
        self.has_run = True
