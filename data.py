import pickle

import numpy as np
from scipy import stats


# # Calculate the real frequency over the linear growth region
# pts = np.array([34.1, 41.0, 48.6, 55.8, 63.0])
# w_avg = (pts[-1] - pts[0]) / (len(pts) - 1)
# print(f"Re(w) over growth: {w_avg:.4f}")

# # Calculate imaginary frequency
# wc = 0.3162
# t = np.array([37.7, 45.2, 52.5, 59.7, 67.1])
# e = np.array([6.42e-6, 1.66e-5, 5.49e-5, 0.000166, 0.000652])
# lr = stats.linregress(t, np.log(e))
# print(f"Im(w) over growth: {lr.slope / 2 / wc:.4f}")
