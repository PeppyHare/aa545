"""Coding project 2: Ideal MHD."""
import time

import numba
import numpy as np
import progressbar
import tables

from mhd1.configuration import (
    Configuration,
    CylindricalConfiguration,
    GridData,
    CylindricalGridData,
)
from mhd1.methods import divB, calc_cfl, linear_mhd_time_step


class MHDModel:
    """Attempt to solve a system under the equations of ideal MHD."""

    has_run = False

    def __init__(self, c: Configuration, check_divB=False):
        """Initialize model."""
        self.c = c
        self.d = GridData(c)
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
        start_time = time.perf_counter()
        for n in numba.prange(c.t_steps):
            # Check for density positivity
            if np.min(d.Q[0]) < 10 ** -12:
                print(f"Density {np.min(d.Q[0])} is too small!")
                break
            time_step_method(
                Q=d.Q,
                dx=c.dx,
                dy=c.dy,
                dz=c.dz,
                dt=c.dt,
            )
            if n % c.subsample_ratio == 0 and n > 0:
                # Write current state to disk
                self.write_out_data(d.Q)

                # Check for CFL condition
                (cfl_x, cfl_y, cfl_z) = calc_cfl(d.Q, c.dx, c.dy, c.dz, c.dt)
                # print(
                #     f"Current CFL numbers: Cx={cfl_x:.3f}, Cy={cfl_y:.3f},"
                #     f" Cz={cfl_z:.3f}"
                # )
                if cfl_x >= 1 or cfl_y >= 1 or cfl_z >= 1:
                    print("CFL condition violated!")
                    break

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
                # print(f"max_divB: {self.d.max_divB[n]}")
            if showprogress:
                bar.update(n)
        if showprogress:
            bar.finish()
        end_time = time.perf_counter()
        print(
            f"Done! Took {(end_time - start_time):.2f}s. Timeseries"
            f" data saved to {self.d.h5_filename}"
        )
        self.has_run = True


class LinearMHDModel:
    """Solve the time-dependent linear MHD equations."""

    has_run = False

    def __init__(self, c: CylindricalConfiguration, check_divB=False):
        """Initialize model."""
        self.c = c
        self.d = CylindricalGridData(c)
        self.check_divB = check_divB
        self.d.prev_v = self.d.v.copy()
        self.d.prev_b = self.d.b.copy()
        self.d.prev_p = self.d.p.copy()

    def run(self, showprogress=True):
        """Run the simulation."""
        c = self.c
        d = self.d
        # leapfrog single-step starting problem
        # print("Compiling methods...", end=None)
        # start_time = time.perf_counter()
        # linear_mhd_time_step(
        #     # next_v=d.prev_v,
        #     # next_b=d.prev_b,
        #     # next_p=d.prev_p,
        #     v=d.v,
        #     b=d.b,
        #     p=d.p,
        #     dr=c.dr,
        #     dz=c.dz,
        #     dt=0,
        #     p0=c.p0,
        #     rho0=c.rho0,
        #     b0r=c.b0r,
        #     b0t=c.b0t,
        #     b0z=c.b0z,
        #     db0rdr=c.db0rdr,
        #     db0rdz=c.db0rdz,
        #     db0tdr=c.db0tdr,
        #     db0tdz=c.db0tdz,
        #     db0zdr=c.db0zdr,
        #     db0zdz=c.db0zdz,
        # )
        # end_time = time.perf_counter()
        # print(f"Finished in {(end_time - start_time)*1000:.1f}ms.")
        if showprogress:
            print(f"Simulating {c.t_steps} time steps ...")
            bar = progressbar.ProgressBar(
                maxval=c.t_steps,
                widgets=[
                    progressbar.Bar("=", "[", "]"),
                    " ",
                    progressbar.Percentage(),
                ],
            )
            bar.start()
        start_time = time.perf_counter()
        for n in range(1, c.t_steps):
            linear_mhd_time_step(
                # next_v=d.v,
                # next_b=d.b,
                # next_p=d.p,
                v=d.v,
                b=d.b,
                p=d.p,
                dr=c.dr,
                dz=c.dz,
                dt=d.dt,
                p0=c.p0,
                rho0=c.rho0,
                b0r=c.b0r,
                b0t=c.b0t,
                b0z=c.b0z,
                db0rdr=c.db0rdr,
                db0rdz=c.db0rdz,
                db0tdr=c.db0tdr,
                db0tdz=c.db0tdz,
                db0zdr=c.db0zdr,
                db0zdz=c.db0zdz,
            )
            d.current_time += d.dt
            # d.v, d.prev_v = d.prev_v, d.v
            # d.b, d.prev_b = d.prev_b, d.b
            # d.p, d.prev_p = d.prev_p, d.p
            if n % c.subsample_ratio == 0 and n > 0:
                # Write current state to disk
                d.write_out_data()

                # Check for CFL condition
                # (cfl_x, cfl_y, cfl_z) = calc_cfl(d.Q, c.dx, c.dy, c.dz, c.dt)
                # print(
                #     f"Current CFL numbers: Cx={cfl_x:.3f}, Cy={cfl_y:.3f},"
                #     f" Cz={cfl_z:.3f}"
                # )
                # if cfl_x >= 1 or cfl_y >= 1 or cfl_z >= 1:
                #     print("CFL condition violated!")
                #     break

            # # Calculate total energy
            # self.d.TE[n] += np.sum(d.Q[7])
            # # Calculate total kinetic energy
            self.d.KE[n] += np.sum(
                (
                    np.real(d.v[0]) ** 2 * c.rho0
                    + np.real(d.v[1]) ** 2 * c.rho0
                    + np.real(d.v[2]) ** 2 * c.rho0
                )
            )
            # # Calculate total magnetic field energy
            # self.d.FE[n] += np.sum(d.Q[4] ** 2 + d.Q[5] ** 2 + d.Q[6] ** 2) / (
            #     2 * c.mu
            # )
            # Check the maximum divergence of B
            # if self.check_divB:
            #     self.d.max_divB[n] += np.max(
            #         np.abs(
            #             divB(
            #                 d.Q[4:7],
            #                 dx=c.dx,
            #                 dy=c.dy,
            #                 dz=c.dz,
            #                 bcx=c.bcx,
            #                 bcy=c.bcy,
            #                 bcz=c.bcz,
            #             )
            #         )
            #     )
            # print(f"max_divB: {self.d.max_divB[n]}")
            if showprogress:
                bar.update(n)
        if showprogress:
            bar.finish()
        end_time = time.perf_counter()
        print(
            f"Done! Took {(end_time - start_time):.2f}s. Timeseries"
            f" data saved to {self.d.h5_filename}"
        )
        self.has_run = True
