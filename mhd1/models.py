"""Coding project 2: Ideal MHD."""
import numba
import numpy as np
import progressbar
import tables

from mhd1.configuration import Configuration, ParticleData
from mhd1.methods import divB


class MHDModel:
    """Attempt to solve a system under the equations of ideal MHD."""

    has_run = False

    def __init__(self, c: Configuration, check_divB=False):
        """Initialize model."""
        self.c = c
        self.d = ParticleData(c)
        self.check_divB = check_divB

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
            if np.min(d.Q[0]) < 10 ** -6:
                raise Exception(f"Density {np.min(d.Q[0])} is too small!")
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
            # Calculate total energy
            self.d.TE[n] += np.sum(d.Q[7])
            # Calculate total kinetic energy
            self.d.KE[n] += np.sum(
                (d.Q[1] ** 2 + d.Q[2] ** 2 + d.Q[3] ** 2) / (2 * d.Q[0])
            )
            # Calculate total magnetic field energy
            self.d.FE[n] += np.sum(d.Q[4] ** 2 + d.Q[5] ** 2 + d.Q[6] ** 2) / (
                2 * c.mu
            )
            # Check the maximum divergence of B
            if self.check_divB:
                self.d.max_divB[n] += np.max(
                    np.abs(
                        divB(
                            d.Q[4:7],
                            dx=c.dx,
                            dy=c.dy,
                            dz=c.dz,
                            bcx=c.bcx,
                            bcy=c.bcy,
                            bcz=c.bcz,
                        )
                    )
                )
                print(f"max_divB: {self.d.max_divB[n]}")
            if showprogress:
                bar.update(n)
        if showprogress:
            bar.finish()
        print(f"Done! Timeseries data saved to {self.d.h5_filename}")
        self.has_run = True
