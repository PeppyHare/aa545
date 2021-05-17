"""Run Ideal MHD Solver."""
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import tables
from functools import partial
from matplotlib.animation import FuncAnimation

from mhd1.configuration import Configuration
from mhd1.models import MHDModel
import mhd1.plots as plots
from mhd1.utils import save_plot


plt.style.use("dark_background")

c = Configuration()
m = MHDModel(c)

plots.plot_initial_distribution(m)
# m.run()


# def _animate_frame(frame, q, X, Y, cf, time_text):
#     Z = q[frame, :, :]
#     cf = ax.contourf(X, Y, Z)
#     time_text.set_text(f"t = {frame}")
#     return (cf, time_text)


# x = c.x_i
# y = c.y_j
# X, Y = np.meshgrid(x, y)

# f = tables.open_file("saved_data/mhd1/2021-05-16_81995.57278_data.h5")
# mx = f.root.mx[:, :, :, 50]
# Z = mx[0, :, :]
# fig, ax = plt.subplots()
# ax.axis("equal")
# ax.axis("off")
# cf = ax.contourf(X, Y, Z)
# # get the mappable, the 1st and the 2nd are the x and y axes
# plt.colorbar(cf, ax=ax)
# time_text = plt.text(0.02, 0.95, "")
# animate = partial(_animate_frame, X=X, Y=Y, q=mx, cf=cf, time_text=time_text)
# animation = FuncAnimation(
#     fig, animate, frames=range(mx.shape[0]), interval=5, repeat=True
# )
# animation.save(
#     "test.mp4",
#     fps=24,
#     progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
# )
# plt.show()
