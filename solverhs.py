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
if dt > 0.5 * dx:
    raise ValueError("CFL condition violated: Decrease dt or increase dx.")

# Spatial and Time Grids
x = np.linspace(0, 150, nx)
t = np.linspace(0, T, nt)

# Initialize Fields
phi = np.zeros((nx, nt), dtype=complex)
pi = np.zeros((nx, nt), dtype=complex)

# Initial Conditions
phi[:, 0] = 5*np.exp(-0.5 * (x / 10)**2)
pi[:, 0] = np.zeros_like(phi[:, 0])

def laplacian(phi, dx=dx):
    """Computes the Laplacian with boundary conditions."""
    phidash = np.zeros_like(phi, dtype=complex)
    for index in range(nx):
        left = index - 1
        right = index + 1
        if index == 0:
            phidash[index] = (6 / dx**2) * (phi[1] - phi[0])  # L'Hôpital's Rule
        elif index == nx - 1:
            d2phidr2 = (2 * phi[index] - 5 * phi[index - 1] + 4 * phi[index - 2] - phi[index - 3]) / dx**2
            #dphidr = (3 * phi[index] - 4 * phi[index - 1] + phi[index - 2]) / (2 * dx)
            #phidash[index] = d2phidr2 + (2 / x[index]) * dphidr
            phidash[index] = (6 / dx**2) * (phi[-2] - phi[-1])
        else:
            phidash[index] = (phi[right] - 2 * phi[index] + phi[left]) / dx**2 + (2 / x[index]) * (phi[right] - phi[left]) / (2 * dx)
    return phidash

def koterm(v, dx=dx):
    """Kreiss-Oliger dissipation term."""
    term = np.zeros_like(v)
    for index in range(2, nx - 2):  # Avoid boundary regions
        term[index] = (v[index+2] - 4*v[index+1] + 6*v[index] - 4*v[index-1] + v[index-2]) / dx
    #term[:2]=term[2]
    #term[-2:]=term[-3]
    return -1*term * (0.02 / 16)

def delr(v,dx=dx):
    derivative=np.zeros_like(v)
    for index in range(1,nx-1):
        derivative[index]=(v[index+1]-v[index-1])/(2*dx)
    derivative[0]=(-3 * v[0] + 4 * v[1] - v[2]) / (2 * dx)
    derivative[-1]=(3 * v[-1] - 4 * v[-2] + v[-3]) / (2 * dx)
    return derivative

def rhs2(a,alpha,vpi,vphidot,i):
    if(i!=0):
        dadt=a*(-((a**2)-1)/(2*x[i])+2*np.pi*x[i]*(((a/alpha)**2)*vpi**2 +vphidot**2))
        #dalphadt=alpha*(+((a**2)-1)/(2*x[i])+2*np.pi*x[i]*(((a/alpha)**2)*vpi**2 +vphidot**2))
        dalphadt=alpha*((dadt/a)+(((a**2) -1)/x[i]))
    else:
        dadt=(2*np.pi*x[i]*(((a/alpha)**2)*vpi**2 +vphidot**2))(a/(a**2+1))
        dalphadt=alpha*()
    return np.array([dadt,dalphadt])
def rhs(pi, phi):
    """Computes right-hand side of wave equation."""
    a=np.random.random(size=x.shape)
    alpha=np.random.random(size=x.shape)
    for i in range(1,nx - 1):

        k1v = rhs2(a[i],alpha[i],pi[i], delr(phi)[i],i)
        k2v = rhs2(a[i] + 0.5 * k1v[0] * dx, alpha[i] + 0.5 * k1v[1] * dx,pi[i], delr(phi)[i],i)
        k3v = rhs2(a[i] + 0.5 * k2v[0] * dx, alpha[i] + 0.5 * k2v[1] * dx,pi[i], delr(phi)[i],i)
        k4v = rhs2(a[i] + k3v[0] * dx, alpha[i] + k3v[1] * dx,pi[i], delr(phi)[i],i)
        a[i+1] = pi[i] + (1/6) * (k1v[0] + 2*k2v[0] + 2*k3v[0] + k4v[0]) * dx
        alpha[i+1] = phi[i] + (1/6) * (k1v[1] + 2*k2v[1] + 2*k3v[1] + k4v[1]) * dx
    dpidt = laplacian(phi)*(alpha/a)**2 +delr(alpha/a)*(alpha/a)*delr(phi)
    dphidt = pi 
    dpidt[-1]= 1*(-3*pi[-1]+4*pi[-2]-pi[-3])/(2*dx) #-(pi[-1]/x[-1])
    print(a)
    return np.array([dpidt, dphidt])

# Time Evolution Using 4th Order Runge-Kutta Method
for i in range(nt - 1):

    k1 = rhs(pi[:, i], phi[:, i])
    k2 = rhs(pi[:, i] + 0.5 * k1[0] * dt, phi[:, i] + 0.5 * k1[1] * dt)
    k3 = rhs(pi[:, i] + 0.5 * k2[0] * dt, phi[:, i] + 0.5 * k2[1] * dt)
    k4 = rhs(pi[:, i] + k3[0] * dt, phi[:, i] + k3[1] * dt)
    pi[:, i+1] = pi[:, i] + (1/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) * dt
    phi[:, i+1] = phi[:, i] + (1/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) * dt

# Visualization Setup
fig, ax = plt.subplots()
line_real, = ax.plot(x, np.real(phi[:, 0]), color="blue", label="Re(phi)")
line_imag, = ax.plot(x, np.imag(phi[:, 0]), color="red", label="Im(phi)")
ax.set_xlim(0, 100)
ax.set_ylim(-5, 5)
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.set_title("Wave Equation Solution")
ax.legend()

def update(frame):
    line_real.set_ydata(np.real(phi[:, frame]))
    line_imag.set_ydata(np.imag(phi[:, frame]))
    ax.set_title(f"Boxphi=eta(delphi)(delphi) at t={t[frame]:.2f} (Without Dissipation)")
    return line_real, line_imag

# Animation
ani = FuncAnimation(fig, update, interval=10, frames=range(0, len(t), 6), repeat_delay=10000)
plt.show()
ani.save("spherical_1.mp4")