
import time

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from configuration import Configuration
from model import PicModel
import plots
from util import save_plot
from weighting import weight_particles


class TestConfiguration(Configuration):
    plot_grid_lines = True
    wp = 1
    max_history_steps = 1000
    markersize = 4

    def initialize_particles(self):
        initial_x = np.linspace(self.x_min, self.x_max)
        v0 = self.beam_velocity
        dx = self.perturbation
        beam1_x = np.linspace(self.x_min, self.x_max, int(self.N / 2) + 1)[:-1]
        beam1_x += dx * np.sin(self.k * beam1_x)
        beam2_x = np.linspace(self.x_min, self.x_max, int(self.N / 2) + 1)[:-1]
        beam2_x -= dx * np.sin(self.k * beam2_x)
        beam1_v = v0 * np.ones_like(beam1_x)
        beam2_v = -v0 * np.ones_like(beam2_x)
        self.initial_x = np.concatenate([beam1_x, beam2_x])
        self.initial_v = np.concatenate([beam1_v, beam2_v])