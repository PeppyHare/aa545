import numpy as np
from matplotlib import pyplot as plt

m = 32
k = 1
L = 1
dx = L / m
grid_pts = np.linspace(0, L, m + 1)[:-1]
test_rho = np.sin(k * 2 * np.pi / L * grid_pts)
soln = test_rho / ((k * 2 * np.pi / L) ** 2)
print(f"test_rho: {test_rho}")
# plt.plot(grid_pts, test_rho)
# plt.show()
rho_fft = np.fft.fft(test_rho)
print(f"rho_fft: {rho_fft}")
R0 = rho_fft[0]
print(f"R0: {R0}")
kx = np.fft.fftfreq(m) / dx
print(f"kx: {kx}")
phi_fft = -0.5 * rho_fft / ((np.cos(2.0 * np.pi * kx / (m)) - 1.0) / dx ** 2)
print(phi_fft)
phi_fft[0] = R0
solved = np.real(np.fft.ifft(phi_fft))
print(f"solved: {solved}")
print(f"soln: {soln}")
print(f"solved - soln: {solved - soln}")
plt.figure()
plt.suptitle("solved")
plt.plot(solved, label="solved")
plt.plot(soln, label="solution")
plt.legend()
plt.show()
# plt.plot(k_vec, rho_fft)
# plt.show()
