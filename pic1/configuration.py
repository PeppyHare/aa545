import math
import random

import numpy as np

from pic1.poisson import setup_poisson


class Configuration:
    """Default configuration for PIC model.

    Configuration parameters in this struct do not change over the course of
    the simulation.
    """

    # Total number of particles
    N = 512

    # Number of grid cells (number of grid points - 1)
    M = 128

    # Left-hand boundary of periodic domain
    x_min = -np.pi

    # Right-hand boundary of periodic domain
    x_max = np.pi

    # Min bound for plotting velocity
    vx_min = -2
    vy_min = -2

    # Max bound for plotting velocity
    vx_max = 2
    vy_max = 2

    # Normalization of electric constant
    eps0 = 1.0

    # Charge-to-mass ratio of species
    qm = -1

    # Plasma frequency
    wp = 1

    # Cyclotron freq q*B_z/m. 0 for B_z = 0
    wc = 0

    # Time step
    dt = 0.01

    # Number of periods of the plasma frequency to integrate in time
    n_periods = 20

    # Weighting order.  0 = nearest grid point. 1 = linear weighting
    weighting_order = 1

    # Should the animation repeat?
    repeat_animation = False

    # Particle plotting size. 1 is small, 20 is large.
    markersize = 2

    # Whether to plot grid lines
    plot_grid_lines = False

    # If false, disable live plots of energy
    plot_energy = False

    # If false, disable live plots of electric field
    plot_fields = False

    # Max number of time steps to hold in memory.
    # If t_steps greater than this, subsampling will occur.
    max_history_steps = 5000

    def __init__(self):
        """Return a valid configuration for PICModel."""
        # Maximum value of time we will integrate to
        self.t_max = self.n_periods * (2 * np.pi / self.wp)

        # Particle charge
        self.q = (self.wp ** 2) * self.eps0 / (self.N * self.qm)

        # Background charge density
        self.rho_bg = -self.N * self.q

        # Grid points
        self.x_j = np.linspace(0, 1, self.M + 1)[:-1]

        # Set the initial position/velocity of all particles.
        self.set_initial_conditions()

        # Set phase space ranges used in plots
        self.x_range = (self.x_min, self.x_max)
        self.vx_range = (self.vx_min, self.vx_max)
        self.vy_range = (self.vy_min, self.vy_max)

        # Scale position from [x_min, x_max] to [0, 1]
        # x' = (x - x_min)/(x_max - x_min)
        self.L = self.x_max - self.x_min

        # Total number of time steps
        self.t_steps = math.ceil(self.t_max / self.dt)

        # Time axis used for plots
        self.time_axis = np.linspace(0, self.t_max, self.t_steps)

        # Grid spacing / particle size
        self.dx = 1 / self.M

        # Particle mass
        self.m = self.q / self.qm

        # Number of subsampled frames to hold in memory
        self.history_steps = min(self.t_steps, self.max_history_steps)

        # State of all particles is stored every {subsample_ratio} time steps
        self.subsample_ratio = math.ceil(self.t_steps / self.history_steps)

        # Finite difference matrix used to solve Poisson equation
        (self.inv_a, _) = setup_poisson(self.M)

    def set_initial_conditions(self):
        """Set the initial position of all particles in phase space.

        Positions/velocities here are absolute, and not yet scaled to [0,1].
        """
        # Default: Maxwellian beam
        v_fwhm = 2
        v_stdev = v_fwhm / 2.355
        random.seed("not really random")
        initial_x = np.zeros(self.N)
        initial_vx = np.zeros(self.N)
        initial_vy = np.zeros(self.N)
        for i in range(self.N):
            initial_x[i] = random.uniform(self.x_min, self.x_max)
            initial_vx[i] = max(
                min(random.gauss(0, v_stdev), self.vx_max), self.vx_min
            )
        self.initial_x = initial_x
        self.initial_vx = initial_vx
        self.initial_vy = initial_vy


class ParticleData:
    """Struct containing the state and history of simulation data.

    The data contained in these objects & vectors will change over the course
    of the simulation.
    """

    def __init__(self, c: Configuration):
        # Calculate total kinetic energy at each time step
        self.ke_hist = np.zeros(c.t_steps)

        # Calculate total electric field energy at each time step
        self.fe_hist = np.zeros(c.t_steps)

        # Calculate total momentum at each time step
        self.p_hist = np.zeros(c.t_steps)

        # Current position of each particle
        self.x_i = np.zeros(c.N)

        # Current velocity of each particle
        self.vx_i = np.zeros(c.N)
        self.vy_i = np.zeros(c.N)

        # Current value of electric field at grid points
        self.ex_j = np.zeros(c.M)

        # Maximum value of total energy seen so far
        self.energy_max = 0

        # Maximum value of electric field seen so far
        self.ex_max = 0

        # Subsampled history of particle positions
        self.x_hist = np.zeros((c.N, c.history_steps + 1))

        # Subsampled history of particle velocities
        self.vx_hist = np.zeros((c.N, c.history_steps + 1))
        self.vy_hist = np.zeros((c.N, c.history_steps + 1))

        # Subsampled history of electric field
        self.ex_hist = np.zeros((c.M, c.history_steps + 1))
        self.ey_hist = np.zeros((c.M, c.history_steps + 1))

        # Subsampled history of particle density
        self.rho_hist = np.zeros((c.M, c.history_steps + 1))