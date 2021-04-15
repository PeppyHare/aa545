import math
import os
from pathlib import Path
import random
import time

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
import numpy as np
import progressbar


t_steps = None

@numba.jit(nopython=True, cache=False)
def time_step(x, frame):
    for i in numba.prange(x.size):
        x[i] += frame / (i + 1.0)
    return x


def run():
    """Run the simulation."""
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for frame in range(t_steps):
        time_step(x, frame)
        bar.update(frame + 1)
    bar.finish()
    print("done!")


def run_slow():
    """Run the simulation."""
    bar = progressbar.ProgressBar(
        maxval=t_steps,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for frame in range(t_steps):
        time_step.py_func(x, frame)
        bar.update(frame + 1)
    bar.finish()
    print("done!")

# Performance testing: print execution time per time_step(), excluding JIT
# compilation time
n = 10000
# Run one short iteration to compile time_step()
t_steps = 1
x = np.zeros(n)
run()
# Then we can run the full simulation without counting compile time
t_steps = 1000
x = np.zeros(n)
start_time = time.perf_counter()
run()
end_time = time.perf_counter()
start_time_slow = time.perf_counter()
run_slow()
end_time_slow = time.perf_counter()
print(
    f"(numba)  Total elapsed time per step (n={n}):"
    f" {1000.0*(end_time - start_time)/t_steps:.5f} ms"
)
print(
    f"(python) Total elapsed time per step (n={n}):"
    f" {1000.0*(end_time_slow - start_time_slow)/t_steps:.5f} ms"
)
print(f"numba speedup: {(end_time_slow - start_time_slow)/(end_time - start_time):.2f} times faster")