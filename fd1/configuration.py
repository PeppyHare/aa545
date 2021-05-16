import math

import numpy as np

from methods import time_step_ftcs_dirichlet


class Configuration:
    """Default configuration for finite difference model.

    Configuration parameters in this struct do not change over the course of
    the simulation.
    """

    # Number of grid points
    M = 201

    # Left-hand boundary of domain
    x_min = 0

    # Right-hand boundary of domain
    x_max = M + 1

    # Time step
    dt = 0.5

    # Advection speed
    c = 1.0

    # Max time
    t_max = M / dt

    # Integration method
    time_step_method = staticmethod(time_step_ftcs_dirichlet)

    # Max number of time steps to hold in memory.
    # If t_steps greater than this, subsampling will occur.
    max_history_steps = 5000

    def __init__(self):
        """Return a valid configuration for AdvectionModel."""
        # Grid points
        self.x_j = np.linspace(self.x_min, self.x_max, self.M)

        # Grid spacing
        self.dx = (self.x_max - self.x_min) / (self.M - 1)

        # Total number of time steps
        self.t_steps = math.ceil(self.t_max / self.dt)

        # Time axis used for plots
        self.time_axis = np.linspace(0, self.t_max, self.t_steps)

        # Number of subsampled frames to hold in memory
        self.history_steps = min(self.t_steps, self.max_history_steps)

        # State of all particles is stored every {subsample_ratio} time steps
        self.subsample_ratio = math.ceil(self.t_steps / self.history_steps)
        print(f"Courant number: {self.c*self.dt/self.dx}")

        # Set the initial state
        self.set_initial_conditions()

    def set_initial_conditions(self):
        """Set the initial conditions to be evolved in time."""
        # Default: Square pulse
        u = np.zeros(self.M)
        u[10:21] = 1
        self.initial_u = u


class ParticleData:
    """Struct containing the state and history of simulation data.

    The data contained in these objects & vectors will change over the course
    of the simulation.
    """

    def __init__(self, c: Configuration):
        """Initialize ParticleData."""
        # The current solution state at each grid point
        self.u_j = np.zeros_like(c.initial_u)
        self.u_j += c.initial_u
        self.u_exact = np.zeros_like(c.initial_u)
        self.u_exact += c.initial_u

        # Store the history of the solution over time
        self.u_hist = np.zeros((c.M, c.max_history_steps))
        self.u_hist[:, 0] += c.initial_u
        self.u_exact_hist = np.zeros((c.M, c.max_history_steps))
        self.u_exact_hist[:, 0] += c.initial_u

        # Solution error
        self.err = np.zeros(c.t_steps)

        # Store the max value of u
        self.u_max = np.zeros(c.t_steps)
