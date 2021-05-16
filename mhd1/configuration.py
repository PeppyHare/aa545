import math

import numpy as np

from fd1.methods import time_step_ftcs_dirichlet


class Configuration:
    """Default configuration for finite difference model.

    Configuration parameters in this struct do not change over the course of
    the simulation.
    """

    # Number of grid points in x
    Mx = 201

    # Number of grid points in y
    My = 201

    # Number of grid points in z
    Mz = 201

    # Left-hand boundary of domain
    x_min = 0
    y_min = 0
    z_min = 0

    # Right-hand boundary of domain
    x_max = Mx + 1
    y_max = My + 1
    z_max = Mz + 1

    # Time step
    dt = 0.5

    # Adiabatic index gamma
    g = 5/3

    # Max time
    t_max = 100

    # Integration method
    time_step_method = staticmethod(time_step_ftcs_dirichlet)

    # Max number of time steps to hold in memory.
    # If t_steps greater than this, subsampling will occur.
    max_history_steps = 5000

    def __init__(self):
        """Return a valid configuration for AdvectionModel."""
        # Grid points
        self.x_i = np.linspace(self.x_min, self.x_max, self.Mx)
        self.y_j = np.linspace(self.y_min, self.y_max, self.My)
        self.z_k = np.linspace(self.z_min, self.z_max, self.Mz)

        # Grid spacing
        self.dx = (self.x_max - self.x_min) / (self.Mx - 1)
        self.dy = (self.y_max - self.y_min) / (self.My - 1)
        self.dz = (self.z_max - self.z_min) / (self.Mz - 1)

        # Total number of time steps
        self.t_steps = math.ceil(self.t_max / self.dt)

        # Time axis used for plots
        self.time_axis = np.linspace(0, self.t_max, self.t_steps)

        # Number of subsampled frames to hold in memory
        self.history_steps = min(self.t_steps, self.max_history_steps)

        # Full state is stored every {subsample_ratio} time steps
        self.subsample_ratio = math.ceil(self.t_steps / self.history_steps)

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
