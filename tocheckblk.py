import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

from critical_collapse_prq import wave_solver,animate_wave

def binarysearch(blkhol_fn, a, b, tol=1e-10):
    while np.abs(b - a) > tol:
        mid = (a + b) / 2
        if blkhol_fn(mid):
            b = mid
        else:
            a = mid
    return mid

def make_blkhol2(dx, dt):
    def blkhol2(amp):
        _, _, phi, a, alpha = wave_solver(dx=dx, dt=dt, p=amp)
        if ((np.isnan(phi).any() or (np.abs(phi[0, :]) > 100 * amp).any()) and (np.isnan(a).any())):
            return True
        else:
            return False
    return blkhol2

def make_blkholbm(dx, dt):
    def blkholbm(amp):
        _, _, phi, a, alpha = wave_solver(dx=dx, dt=dt, p=amp)
        if np.min(alpha) == 0:
            return True
        elif np.linalg.norm(alpha - np.ones_like(alpha)) < 1e-15:
            return False
        else:
            return True  # intermediate case: treat as black hole
    return blkholbm

# List of spatial resolutions
resolutions = [4.0, 2.0, 1.0, 0.5]
dt = 0.1
results = []

for dx in resolutions:
    print(f"Running for dx = {dx}")
    p_blkhol2 = binarysearch(make_blkhol2(dx, dt), 0.001, 1.0)
    p_blkholbm = binarysearch(make_blkholbm(dx, dt), 0.001, 1.0)
    diff = abs(p_blkhol2 - p_blkholbm)
    results.append((dx, p_blkhol2, p_blkholbm, diff))
    print(f"dx = {dx:.2f}, blkhol2 p* = {p_blkhol2:.12f}, blkholbm p* = {p_blkholbm:.12f}, diff = {diff:.2e}\n")

# Optionally, plot convergence
import matplotlib.pyplot as plt

dxs = [r[0] for r in results]
diffs = [r[3] for r in results]

plt.figure(figsize=(8,5))
plt.loglog(dxs, diffs, 'o-', label='|p*_blkhol2 - p*_blkholbm|')
plt.xlabel("dx (log scale)")
plt.ylabel("Difference in critical p* (log scale)")
plt.title("Convergence of Critical Amplitude Estimates")
plt.grid(True, which='both', ls='--')
plt.legend()
plt.show()
