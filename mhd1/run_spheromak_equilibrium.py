"""Run Ideal MHD Solver."""
import math
import itertools
import os

import numpy as np
from matplotlib import pyplot as plt

from mhd1.configuration import CylindricalConfiguration
from mhd1.models import LinearMHDModel
from mhd1.utils import save_data, load_data
import mhd1.plots as plots


plt.style.use("dark_background")  # For comfortable nighttime coding
demo_mode = True


class SpheromakConfiguration(CylindricalConfiguration):
    """Linearized m=1 Spheromak equilibrium."""

    def __init__(self, R=1.0, L=1.0):
        self.R = R
        self.L = L
        CylindricalConfiguration.__init__(self)


c = SpheromakConfiguration(R=1.0, L=1.0)
m = LinearMHDModel(c)
m.run()

d = m.d
print(f"v:{d.v}")
plt.figure()

# plots.mhd_snapshot(m, n=5)
# plots.shock_snapshots(m, n=0)
# plots.plot_shock(m)


# if demo_mode:
#     m = load_data(
#         os.path.join(
#             "saved_data", "mhd1", "2021-05-23_83581.340669_screw_pinch_latest.p"
#         )
#     )
# else:
#     c = ScrewPinchConfiguration(
#         Mx=65, My=65, Mz=65, dt=0.05, t_max=10, j0=0.5, R=8
#     )
#     m = MHDModel(c, check_divB=True)
#     plots.plot_initial_distribution_all_axes(m)

#     m.run()
#     save_data(m, "screw_pinch_latest.p")


# plots.mhd_snapshot(m, 0)
# plots.animate_mhd(m)
