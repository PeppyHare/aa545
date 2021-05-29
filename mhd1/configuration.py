import datetime
import os
import math

import numpy as np
import tables

from mhd1.methods import maccormack_time_step
from mhd1.utils import create_folder


class Configuration:
    """Default configuration for finite difference model.

    Configuration parameters in this struct do not change over the course of
    the simulation.
    """

    # Number of grid points in x
    Mx = 9

    # Number of grid points in y
    My = 9

    # Number of grid points in z
    Mz = 9

    # Left-hand boundary of domain
    x_min = 0
    y_min = 0
    z_min = 0

    # Right-hand boundary of domain
    x_max = Mx - 1
    y_max = My - 1
    z_max = Mz - 1

    # Time step
    dt = 0.5

    # Adiabatic index
    gamma = 5 / 3

    # Vacuum permeability
    mu = 1

    # Max time
    t_max = 100

    # Boundary condition types in each direction
    # 0: periodic boundary
    bcx = 0
    bcy = 0
    bcz = 0

    # Integration method
    time_step_method = staticmethod(maccormack_time_step)

    # Max number of time steps to write to disk.
    # If t_steps greater than this, subsampling will occur.
    max_history_steps = 32

    def __init__(self):
        """Return a valid configuration for MHDModel."""
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

        # Number of subsampled frames to write to disk
        self.history_steps = min(self.t_steps, self.max_history_steps)

        # Full state is stored every {subsample_ratio} time steps
        self.subsample_ratio = math.ceil(self.t_steps / self.history_steps)

        # Set the initial state
        self.set_initial_conditions()

    def set_initial_conditions(self):
        """Set the initial conditions to be evolved in time.

        The conserved variables q are stored at each grid point (i, j, k):

        Q[0] = rho
        Q[1] = rho*vx
        Q[2] = rho*vy
        Q[3] = rho*vz
        Q[4] = Bx
        Q[5] = By
        Q[6] = Bz
        Q[7] = e

        Initial configuration:
        """

        Q = np.zeros((8, self.Mx, self.My, self.Mz))
        self.initial_Q = Q


class CylindricalConfiguration:
    """Default 2D configuration for cylindrical finite difference model.

    Configuration parameters in this struct do not change over the course of
    the simulation.
    """

    # Number of grid points in r
    Mr = 101

    # Number of grid points in z
    Mz = 101

    # Scale length of domain
    R = 1
    L = 2

    # Time step
    dt = 0.05

    # Adiabatic index
    gamma = 5 / 3

    # Vacuum permeability
    mu = 1

    # Max time
    t_max = 100

    # Boundary condition types in each direction
    # 0: periodic boundary
    bcx = 0
    bcy = 0
    bcz = 0

    # Integration method
    time_step_method = staticmethod(maccormack_time_step)

    # Max number of time steps to write to disk.
    # If t_steps greater than this, subsampling will occur.
    max_history_steps = 100

    def __init__(self):
        """Return a valid configuration for MHDModel."""
        # Spatial domain boundaries
        self.r_min = 0
        self.r_max = self.R
        self.z_min = 0
        self.z_max = self.L
        # Grid points
        self.r_j = np.linspace(self.r_min, self.r_max)
        self.z_k = np.linspace(self.z_min, self.z_max, self.Mz)

        # Grid spacing
        self.dr = (self.r_max - self.r_min) / (self.Mr - 1)
        self.dz = (self.z_max - self.z_min) / (self.Mz - 1)

        # Total number of time steps
        self.t_steps = math.ceil(self.t_max / self.dt)

        # Time axis used for plots
        self.time_axis = np.linspace(0, self.t_max, self.t_steps)

        # Number of subsampled frames to write to disk
        self.history_steps = min(self.t_steps, self.max_history_steps)

        # Full state is stored every {subsample_ratio} time steps
        self.subsample_ratio = math.ceil(self.t_steps / self.history_steps)

        # Set the initial state
        self.set_initial_conditions()

    def set_initial_conditions(self):
        """Set the initial conditions to be evolved in time.

        v[0] = v1_r
        v[1] = v1_theta
        v[2] = v1_z
        b[0] = B1_r
        b[1] = B1_theta
        b[2] = B1_z
        p = p1

        In additon to the perturbed primitive quantities we will evolve in time,
        we also require the equilibrium state, as well as the spatial gradients
        of the equilibrium quantities:

        b0r = B0_r
        b0t = B0_theta
        b0z = B0_z
        db0rdr = d/dr(B0_r)
        db0rdz = d/dz(B0_r)
        db0tdr = d/dr(B0_theta)
        db0tdz = d/dz(B0_theta)
        db0zdr = d/dr(B0_z)
        db0zdz = d/dz(B0_z)
        p0 = p_0
        rho0 = rho_0

        """

        self.initial_v = np.zeros((3, self.Mr, self.Mz), dtype="cfloat")
        self.initial_b = np.zeros((3, self.Mr, self.Mz), dtype="cfloat")
        self.initial_p = np.zeros((self.Mr, self.Mz), dtype="cfloat")

        self.b0r = np.zeros((self.Mr, self.Mz))
        self.b0t = np.zeros((self.Mr, self.Mz))
        self.b0z = np.zeros((self.Mr, self.Mz))
        self.db0rdr = np.zeros((self.Mr, self.Mz))
        self.db0rdz = np.zeros((self.Mr, self.Mz))
        self.db0tdr = np.zeros((self.Mr, self.Mz))
        self.db0tdz = np.zeros((self.Mr, self.Mz))
        self.db0zdr = np.zeros((self.Mr, self.Mz))
        self.db0zdz = np.zeros((self.Mr, self.Mz))

        self.p0 = np.ones((self.Mr, self.Mz))
        self.rho0 = np.ones((self.Mr, self.Mz))


class GridData:
    """Struct containing the state and history of simulation data.

    The data contained in these objects & vectors will change over the course
    of the simulation.
    """

    def __init__(self, c: Configuration):
        """Initialize GridData."""
        # The current solution state at each grid point
        self.Q = np.copy(c.initial_Q)
        # The total kinetic energy
        self.KE = np.zeros(c.t_steps)
        # The total internal energy
        self.TE = np.zeros(c.t_steps)
        # Total magnetic field energy
        self.FE = np.zeros(c.t_steps)
        # Maximum value of the divergence of B
        self.max_divB = np.zeros(c.t_steps)

        # Store the history of the solution over time in a pytables dataset
        create_folder(os.path.join(os.getcwd(), "saved_data", "mhd1"))
        now_seconds = (
            datetime.datetime.now()
            - datetime.datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        ).total_seconds()
        self.h5_filename = os.path.join(
            os.getcwd(),
            "saved_data",
            "mhd1",
            f"{datetime.datetime.now().strftime('%Y-%m-%d_') + str(now_seconds)}_data.h5",
        )
        max_snapshots = min(c.max_history_steps, c.t_steps)
        with tables.open_file(self.h5_filename, "w") as f:
            atom = tables.Float64Atom()
            q0 = f.create_earray(
                f.root,
                "rho",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q0.append(c.initial_Q[0, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))
            q1 = f.create_earray(
                f.root,
                "mx",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q1.append(c.initial_Q[1, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))
            q2 = f.create_earray(
                f.root,
                "my",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q2.append(c.initial_Q[2, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))
            q3 = f.create_earray(
                f.root,
                "mz",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q3.append(c.initial_Q[3, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))
            q4 = f.create_earray(
                f.root,
                "bx",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q4.append(c.initial_Q[4, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))
            q5 = f.create_earray(
                f.root,
                "by",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q5.append(c.initial_Q[5, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))
            q6 = f.create_earray(
                f.root,
                "bz",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q6.append(c.initial_Q[6, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))
            q7 = f.create_earray(
                f.root,
                "e",
                atom,
                (0, c.Mx, c.My, c.Mz),
                expectedrows=max_snapshots,
            )
            q7.append(c.initial_Q[7, :, :, :].reshape((1, c.Mx, c.My, c.Mz)))


class CylindricalGridData:
    """Struct containing the state and history of simulation data.

    The data contained in these objects & vectors will change over the course
    of the simulation.
    """

    def __init__(self, c: CylindricalConfiguration):
        """Initialize CylindricalGridData."""
        # The current solution state at each grid point
        self.b = np.copy(c.initial_b)
        self.v = np.copy(c.initial_v)
        self.p = np.copy(c.initial_p)

        # The current time
        self.current_time = 0.0

        # TODO: Adaptive time step
        self.dt = c.dt

        # The total kinetic energy
        self.KE = np.zeros(c.t_steps)
        # The total internal energy
        self.TE = np.zeros(c.t_steps)
        # Total magnetic field energy
        self.FE = np.zeros(c.t_steps)
        # Maximum value of the divergence of B
        self.max_divB = np.zeros(c.t_steps)

        # Store the history of the solution over time in a pytables dataset
        create_folder(os.path.join(os.getcwd(), "saved_data", "mhd1"))
        now_seconds = (
            datetime.datetime.now()
            - datetime.datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        ).total_seconds()
        self.h5_filename = os.path.join(
            os.getcwd(),
            "saved_data",
            "mhd1",
            f"{datetime.datetime.now().strftime('%Y-%m-%d_') + str(now_seconds)}_data.h5",
        )
        max_snapshots = min(c.max_history_steps, c.t_steps)
        with tables.open_file(self.h5_filename, "w") as f:
            atom = tables.Float64Atom()
            b_stor = f.create_earray(
                f.root,
                "b",
                atom,
                (0, 3, c.Mr, c.Mz),
                expectedrows=max_snapshots,
            )
            b_stor.append(c.initial_b[:, :].reshape((1, 3, c.Mr, c.Mz)))
            v_stor = f.create_earray(
                f.root,
                "v",
                atom,
                (0, 3, c.Mr, c.Mz),
                expectedrows=max_snapshots,
            )
            v_stor.append(c.initial_v[:, :].reshape((1, 3, c.Mr, c.Mz)))
            p_stor = f.create_earray(
                f.root,
                "p",
                atom,
                (0, c.Mr, c.Mz),
                expectedrows=max_snapshots,
            )
            p_stor.append(c.initial_p.reshape((1, c.Mr, c.Mz)))
            t_stor = f.create_earray(
                f.root,
                "t",
                atom,
                (0,),
                expectedrows=max_snapshots,
            )
            t_stor.append([self.current_time])

    def write_out_data(self):
        Mr = self.v.shape[1]
        Mz = self.v.shape[2]
        with tables.open_file(self.h5_filename, mode="a") as f:
            f.root.v.append(self.v.reshape((1, 3, Mr, Mz)))
            f.root.b.append(self.b.reshape((1, 3, Mr, Mz)))
            f.root.p.append(self.p.reshape((1, Mr, Mz)))
            f.root.t.append([self.current_time])
