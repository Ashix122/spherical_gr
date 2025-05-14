import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid and Time Discretization
nt = 5000  # Number of time steps
nx = 100   # Number of spatial points
T = 300    # Total simulation time
dx = 150 / (nx + 1)
dt = T / (nt + 1)

# CFL Condition for Stability
if dt >  dx:
    raise ValueError("CFL condition violated: Decrease dt or increase dx.")

# Spatial and Time Grids
x = np.linspace(0, 150, nx)
t = np.linspace(0, T, nt)

# Initialize Fields
phi = np.zeros((nx, nt), dtype=complex)
pi = np.zeros((nx, nt), dtype=complex)

# Initial Conditions
phi[:, 0] = 0.001* np.exp(-0.5 * (x / 10)**2)
pi[:, 0] = np.zeros_like(phi[:, 0])

def laplacian(phi, dx=dx):
    phidash = np.zeros_like(phi, dtype=complex)
    for index in range(nx):
        left = index - 1
        right = index + 1
        if index == 0:
            phidash[index] = (6 / dx**2) * (phi[1] - phi[0])
        elif index == nx - 1:
            phidash[index] = (6 / dx**2) * (phi[-2] - phi[-1])
        else:
            phidash[index] = (phi[right] - 2 * phi[index] + phi[left]) / dx**2 +(2 / x[index]) * (phi[right] - phi[left]) / (2 * dx)
    return phidash

def delr(v, dx=dx):
    derivative = np.zeros_like(v)
    for index in range(1, nx - 2 ):
        derivative[index] = (v[index+1] - v[index-1]) / (2 * dx)
    derivative[0] = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * dx)
    derivative[-1] = (3 * v[-1] - 4 * v[-2] + v[-3]) / (2 * dx)
    return derivative

def rhs2(a, alpha, vpi, vphidash, i):
    if i != 0:
        dadt = a * (-((a**2) - 1) / (2 * x[i]) + 2 * np.pi * x[i] * (vpi**2+ vphidash**2))
        dalphadt = alpha * ((dadt / a) + ((a**2 - 1) / x[i]))
    else:
        dadt = 0
        dalphadt = 0
    return np.array([dadt, dalphadt])

def rhs(pi, phi):
    a = np.zeros_like(x)
    alpha = np.zeros_like(x)
    a[0] = 1
    alpha[0] = 1

    for i in range(nx - 1):
        k1v = rhs2(a[i], alpha[i], pi[i], delr(phi)[i], i)
        k2v = rhs2(a[i] + (k1v[0] * dx), alpha[i] + (k1v[1] * dx), pi[i+1], delr(phi)[i+1], i+1)
        a[i + 1] = a[i] + 0.5 * (k1v[0] + k2v[0]) * dx
        alpha[i + 1] = alpha[i] + 0.5 * (k1v[1] + k2v[1]) * dx
    dpidt = laplacian(phi) * (alpha / a)**2 + (delr(alpha) * a - delr(a) * alpha) * delr(phi) / (a**2)
    dphidt = pi * alpha / a
    return np.array([dpidt, dphidt])

# Time Evolution Using 1st Order Runge-Kutta Method (Euler)
for i in range(nt - 1):
    k = rhs(pi[:, i], phi[:, i])
    pi[:, i+1] = pi[:, i] + dt * k[0]
    phi[:, i+1] = phi[:, i] + dt * k[1]

# Visualization Setup
fig, ax = plt.subplots()
line_real, = ax.plot(x, np.real(phi[:, 0]), color="blue", label="Re(phi)")
line_imag, = ax.plot(x, np.imag(phi[:, 0]), color="red", label="Im(phi)")
ax.set_xlim(0, 150)
ax.set_ylim(-0.001, 0.001)
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.set_title("Wave Equation Solution")
ax.legend()

def update(frame):
    line_real.set_ydata(np.real(phi[:, frame]))
    line_imag.set_ydata(np.imag(phi[:, frame]))
    ax.set_title(f"Boxphi=eta(delphi)(delphi) at t={t[frame]:.2f} (RK1 Time Integration)")
    return line_real, line_imag

# Animation
ani = FuncAnimation(fig, update, interval=10, frames=range(0, len(t), 6), repeat_delay=10000)
plt.show()
