import numpy as np
import numba
import time
import h5py

# with h5py.File("data.hdf5", "w") as f:
#     dset = f.create_dataset(
#         "Q_hist", (8, 201, 201, 201, 100), dtype="f", compression="gzip"
#     )

# with h5py.File("data.hdf5", "r") as f:
#     print("reading")
#     start_time = time.perf_counter()
#     print(f["Q_hist"][:, 0, 0, 0, 0])
#     end_time = time.perf_counter()
#     print(f"Total elapsed time: {10**6 * (end_time - start_time):.3f} µs")

# with h5py.File("data.hdf5", "a") as f:
#     print("appending")
#     start_time = time.perf_counter()
#     f["Q_hist"][:, :, :, :, 2] = np.ones((8, 201, 201, 201))
#     end_time = time.perf_counter()
#     print(f"Total elapsed time: {10**6 * (end_time - start_time):.3f} µs")

# with h5py.File("data.hdf5", "r") as f:
#     print("reading")
#     start_time = time.perf_counter()
#     print(f["Q_hist"][:, 0, 0, 0, 1])
#     end_time = time.perf_counter()
#     print(f"Total elapsed time: {10**6 * (end_time - start_time):.3f} µs")

# import os

# import tables
# import numpy as np

# filename = "outarray.h5"

# os.remove(filename)

# f = tables.open_file(filename, mode="w")
# atom = tables.Float64Atom()

# array_c = f.create_earray(
#     f.root, "rho", atom, (0, 201, 201, 201), expectedrows=200
# )
# array_c = f.create_earray(
#     f.root, "mx", atom, (0, 201, 201, 201), expectedrows=200
# )
# array_c = f.create_earray(
#     f.root, "my", atom, (0, 201, 201, 201), expectedrows=200
# )
# array_c = f.create_earray(
#     f.root, "mz", atom, (0, 201, 201, 201), expectedrows=200
# )

# print(f.root.rho)
# f.close()

# f = tables.open_file(filename, mode="a")
# for i in range(200):
#     start_time = time.perf_counter()
#     f.root.rho.append(i * np.ones((1, 201, 201, 201)))
#     f.root.mx.append(i * np.ones((1, 201, 201, 201)))
#     f.root.my.append(i * np.ones((1, 201, 201, 201)))
#     f.root.mz.append(i * np.ones((1, 201, 201, 201)))
#     end_time = time.perf_counter()
#     print(f"Total elapsed time: {10**6 * (end_time - start_time):.3f} µs")

# print(f.root.rho[:, 0, 0, 0])
# f.close()

from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


plt.style.use("dark_background")


def _animate_frame(frame, time_text):

    global cf, ax
    x = np.arange(128)
    y = np.arange(128)
    X, Y = np.meshgrid(x, y)
    Z = (
        np.sin(np.pi * (X / 16 + 0.01 * frame))
        * np.cos(np.pi * (Y / 32 * frame)) ** 2
    )
    cf = ax.contourf(X, Y, Z)
    time_text.set_text(f"t = {frame}")
    return (cf, time_text)


x = np.arange(128)
y = np.arange(128)
X, Y = np.meshgrid(x, y)
z = np.sin(np.pi * (X / 16)) * np.cos(np.pi * Y / 32) ** 2
fig, ax = plt.subplots()
ax.axis("equal")
ax.axis("off")
cf = ax.contourf(X, Y, z)
time_text = plt.text(0.02, 0.95, "")
animate = partial(_animate_frame, time_text=time_text)
animation = FuncAnimation(
    fig, animate, frames=range(100), interval=5, repeat=True
)
animation.save(
    "test.mp4",
    fps=24,
    progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
)
plt.show()
