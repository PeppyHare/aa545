"""Coding project 2: Finite Difference Integrator."""
import numba
import numpy as np
import progressbar

from fd1.configuration import Configuration, ParticleData
from fd1.methods import exact_advection_periodic


class AdvectionModel:
    """Finite difference solver for the 1D advection equation.

    For the simple model equation, the exact solution is known, so we can
    compute the error at each time step.
    """

    has_run = False

    def __init__(self, c: Configuration):
        """Initialize model."""
        self.c = c
        self.d = ParticleData(c)

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
            exact_advection_periodic(
                u_j=d.u_exact,
                u_hist=d.u_exact_hist,
                n=n,
                subsample_ratio=c.subsample_ratio,
                dx=c.dx,
                dt=c.dt,
                c=c.c,
            )
            time_step_method(
                u_j=d.u_j,
                u_hist=d.u_hist,
                n=n,
                subsample_ratio=c.subsample_ratio,
                dx=c.dx,
                dt=c.dt,
                c=c.c,
            )
            d.err[n] += np.sum((d.u_j - d.u_exact) ** 2)
            d.u_max[n] += np.max(np.abs(d.u_j))

            if showprogress:
                bar.update(n)
        if showprogress:
            bar.finish()
        print("done!")
        self.has_run = True
