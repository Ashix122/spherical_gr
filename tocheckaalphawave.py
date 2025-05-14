import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid and Time Discretization
T = 1500    # Total simulation time
dx = 4.0
dt = 0.1
x = np.arange(0, 400+dx, dx)
t = np.arange(0, T+dt, dt)
nx, nt = len(x), len(t)

# CFL Condition for Stability
if dt > dx:
    raise ValueError("CFL condition violated: Decrease dt or increase dx.")

# Spatial and Time Grids

# Initialize Fields
phi = np.zeros((nx, nt), dtype=complex)
pi = np.zeros((nx, nt), dtype=complex)

# Initial Conditions
phi[:, 0] = 0.001 * np.exp(-0.5 * (x / 10)**2)
pi[:, 0] = np.zeros_like(phi[:, 0])
def koterm(v, dx=dx):
    term = np.zeros_like(v)
    for index in range(2, nx - 2):  # Avoid boundary regions
        term[index] = (v[index+2] - 4*v[index+1] + 6*v[index] - 4*v[index-1] + v[index-2]) / dx
    term[:2]=term[2]
    term[-2:]=term[-3]
    return -1*term * (0.02*10 / 16)

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
            phidash[index] = (phi[right] - 2 * phi[index] + phi[left]) / dx**2 + (2 / x[index]) * (phi[right] - phi[left]) / (2 * dx)
    return phidash

def delr(v, dx=dx):
    derivative = np.zeros_like(v)
    for index in range(1, nx - 1):
        derivative[index] = (v[index+1] - v[index-1]) / (2 * dx)
    derivative[0] = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * dx)
    derivative[-1] = (3 * v[-1] - 4 * v[-2] + v[-3]) / (2 * dx)
    return derivative

def rhs2(a, alpha, vpi, vphidash, i):
    if i != 0:
        dadt = a * (-((a**2) - 1) / (2 * x[i]) + 2 * np.pi * x[i] * (vpi**2 + vphidash**2))
        dalphadt = alpha * ((dadt / a) + ((a**2 - 1) / x[i]))
    else:
        dadt = 0
        dalphadt = 0
    return np.array([dadt, dalphadt])

def rhs(pi, phi,a,alpha):
    
    dpidt = laplacian(phi) * (alpha / a)**2 + (delr(alpha) * a - delr(a) * alpha) * delr(phi) / (a**2) 
    dpidt[-1]= 1*(-3*pi[-1]+4*pi[-2]-pi[-3])/(2*dx)
    dpidt=dpidt +koterm(pi)
    dphidt = pi * (alpha / a) + koterm(phi)
    return np.array([dpidt, dphidt])

# Time Evolution Using 2nd Order Runge-Kutta Method (RK2)

for i in range(nt - 1):
    # Stage 1
    a = np.ones_like(x)
    alpha = np.ones_like(x)
    

    k1 = rhs(pi[:, i], phi[:, i],a,alpha)
    
    # Intermediate state
    pi_half = pi[:, i] + dt * k1[0]
    phi_half = phi[:, i] + dt * k1[1]
    
    # Stage 2
    k2 = rhs(pi_half, phi_half,a,alpha)
    
    # Full RK2 update
    pi[:, i+1] = pi[:, i] + (dt/2) *(k1[0]+k2[0])
    phi[:, i+1] = phi[:, i] + (dt/2) *(k1[1]+k2[1])


# Visualization Setup
fig, ax = plt.subplots()
line_real, = ax.plot(x, np.real(phi[:, 0]), color="blue", label="Re(phi)")
line_imag, = ax.plot(x, np.imag(phi[:, 0]), color="red", label="Im(phi)")
ax.set_xlim(0, 400)
ax.set_ylim(-0.0002, 0.0002)
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.set_title("Wave Equation Solution")
ax.legend()

def update(frame):
    line_real.set_ydata(np.real(phi[:, frame]))
    line_imag.set_ydata(np.imag(phi[:, frame]))
    ax.set_title(f"Spherical GR at t={t[frame]:.2f} (RK2 Time Integration)")
    return line_real, line_imag

# Animation
ani = FuncAnimation(fig, update, interval=100, frames=range(0, len(t), 50), repeat_delay=10000)
plt.show()
ani.save("spherical_gr_withconstata.mp4")
