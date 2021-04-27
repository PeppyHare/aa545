"""Utility functions"""

import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


def create_folder(path):
    """Create a (possibly nested) folder if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    create_folder(os.path.join(os.getcwd(), "plots", "pic1"))
    fig_name = os.path.join("plots", "pic1", filename)
    plt.savefig(fig_name)
    print(f"Saved figure {os.path.join(os.getcwd(), fig_name)} to disk.")


def count_crossings(arr):
    """Count the number of times the value in numpy.array :arr: crosses its
    average value"""
    avg = np.average(arr)
    is_above = arr[0] > avg
    crossings = 0
    for n in range(1, arr.shape[0]):
        if is_above and arr[n] < avg:
            is_above = False
            crossings += 1
        elif (not is_above) and arr[n] > avg:
            is_above = True
            crossings += 1
    return crossings
