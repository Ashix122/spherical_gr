import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

@njit
def laplacian(phi, x, dx):
    nx = len(x)
    phidash = np.zeros_like(phi)
    for index in range(nx):
        if index == 0:
            phidash[index] = (6 / dx**2) * (phi[1] - phi[0])
        elif index == nx - 1:
            phidash[index] = (6 / dx**2) * (phi[-2] - phi[-1])
        else:
            phidash[index] = (phi[index + 1] - 2 * phi[index] + phi[index - 1]) / dx**2
            phidash[index] += (2 / x[index]) * (phi[index + 1] - phi[index - 1]) / (2 * dx)
    return phidash

@njit
def delr(v, dx):
    nx = len(v)
    derivative = np.zeros_like(v)
    for index in range(1, nx - 1):
        derivative[index] = (v[index + 1] - v[index - 1]) / (2 * dx)
    derivative[0] = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * dx)
    derivative[-1] = (3 * v[-1] - 4 * v[-2] + v[-3]) / (2 * dx)
    return derivative

@njit
def rhs2(a, alpha, vpi, vphidash, r, i):
    if i != 0:
        dadt = a * (-((a**2) - 1) / (2 * r) + 2 * np.pi * r * (vpi**2 + vphidash**2))
        dalphadt = alpha * ((dadt / a) + ((a**2 - 1) / r))
    else:
        dadt = 0
        dalphadt = 0
    return dadt, dalphadt

@njit
def koterm(v, dx):
    nx = len(v)
    term = np.zeros_like(v)
    v_ext = np.concatenate((np.array([v[1], v[0]]), v))

    for i in range(nx):
        i_ext = i + 2
        if i >= 2 and i <= nx - 3:
            term[i] = (v_ext[i_ext+2] - 4*v_ext[i_ext+1] + 6*v_ext[i_ext] - 4*v_ext[i_ext-1] + v_ext[i_ext-2]) / dx
        elif i < 2:
            term[i] = (v_ext[i_ext+2] - 4*v_ext[i_ext+1] + 6*v_ext[i_ext] - 4*v_ext[i_ext-1] + v_ext[i_ext-2]) / dx
        else:
            term[i] = 0
    return -1 * term * (0.4/ 16)


@njit
def rhs(pi, phi, a, alpha, x, dx):
    nx = len(x)
    dphidt = np.zeros_like(pi)
    dpidt = np.zeros_like(pi)
    dphi = delr(phi, dx)
    dalpha = delr(alpha, dx)
    da = delr(a, dx)

    lap = laplacian(phi, x, dx)
    for i in range(nx):
        dpidt[i] = lap[i] * (alpha[i] / a[i])**2 + (dalpha[i] * a[i] - da[i] * alpha[i]) * dphi[i] / (a[i]**2)
    dpidt[-1] = ((-3*pi[-1]+4*pi[-2]-pi[-3])/(2*dx))*(alpha/a)[-1] -(pi[-1]/x[-1])*(alpha/a)[-1]
    #dpidt[0]=(alpha[0]/a[0])*(-3*pi[0]+4*pi[1]-pi[2])/(2*dx)
    dpidt += koterm(pi, dx)
    dphidt = pi * (alpha / a) + koterm(phi, dx) 
    return dpidt, dphidt

def wave_solver(dx, dt, T=1500):
    x = np.arange(0, 400+dx, dx)
    t = np.arange(0, T+dt, dt)
    nx, nt = len(x), len(t)

    if dt > dx:
        raise ValueError("CFL condition violated: Decrease dt or increase dx.")

    phi = np.zeros((nx, nt))
    pi = np.zeros((nx, nt))

    phi[:, 0] = 1 * np.exp(-0.5 * (x / 10)**2)
    pi[:, 0] = np.zeros_like(phi[:, 0])

    for i in range(nt - 1):
        a = np.zeros_like(x)
        alpha = np.zeros_like(x)
        a[0] = 1
        alpha[0] = 1

        dphi = delr(phi[:, i], dx)

        for j in range(nx - 1):
            dadt, dalphadt = rhs2(a[j], alpha[j], pi[j,i], dphi[j], x[j], j)
            dadt2, dalphadt2 = rhs2(a[j] + dadt * dx, alpha[j] + dalphadt * dx, pi[j+1,i], dphi[j+1], x[j+1], j+1)
            a[j + 1] = a[j] + 0.5 * (dadt + dadt2) * dx
            alpha[j + 1] = alpha[j] + 0.5 * (dalphadt + dalphadt2) * dx

        dpidt, dphidt = rhs(pi[:, i], phi[:, i], a, alpha, x, dx)
        pi_half = pi[:, i] + dt * dpidt
        phi_half = phi[:, i] + dt * dphidt

        dpidt2, dphidt2 = rhs(pi_half, phi_half, a, alpha, x, dx)

        pi[:, i+1] = pi[:, i] + (dt / 2) * (dpidt + dpidt2)
        phi[:, i+1] = phi[:, i] + (dt / 2) * (dphidt + dphidt2)

    return x, t, phi


def compute_pointwise_self(dx, dt):
    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)

    phi_num_matched = phi_num_fine[::2, ::2]
    phi_num_matched_vfine = phi_num_vfine[::4, ::4]
    phi_num_matched_vvfine = phi_num_vvfine[::8, ::8]

    store = phi_num_matched - phi_num
    store2 = 4 * (phi_num_matched_vfine - phi_num_matched)
    store3 = 16 * (phi_num_matched_vvfine - phi_num_matched_vfine)

    point_plot(store, store2, store3, x, t, "self")

def point_plot(store, store2, store3, x, t, typer):
    fig, ax = plt.subplots()
    if typer == "self":
        a, b, c = "Medium-low", "4*(high-medium)", "16*(higher-high)"
    else:
        a, b, c = "low-exact", "4*Medium-exact", "16*High-exact"

    line_real, = ax.plot(x[:], store[:, 0], color="blue", label=a)
    line_imag, = ax.plot(x[:], store2[:, 0], color="red", label=b)
    line3, = ax.plot(x[:], store3[:, 0], color="green", label=c)

    ax.set_xlim(0, 400)
    ax.set_ylim(-0.000025, 0.000025)
    ax.set_xlabel("x")
    ax.set_ylabel("phi")
    ax.set_title("Wave Equation Solution")
    ax.legend()

    def update(frame):
        line_real.set_ydata(store[:, frame])
        line_imag.set_ydata(store2[:, frame])
        line3.set_ydata(store3[:, frame])
        ax.set_title(f"Wave Equation Solution at t={t[frame]:.4f}")
        return line_real, line_imag, line3

    ani = FuncAnimation(fig, update, interval=50, frames=range(0,len(t),10), repeat_delay=10000)
    plt.show()
    ani.save("pointwise_convergence.mp4")

def compute_norm_self(dx, dt):
    
    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)
    
    phi_num_matched = phi_num_fine[::2, ::2]
    phi_num_matched_vfine = phi_num_vfine[::4, ::4]
    phi_num_matched_vvfine = phi_num_vvfine[::8, ::8]
    
    store = phi_num_matched - phi_num
    store2 = (phi_num_matched_vfine - phi_num_matched)
    store3 = phi_num_matched_vvfine - phi_num_matched_vfine
    error1=np.zeros_like(t)
    error2=np.zeros_like(t)
    error3=np.zeros_like(t)

    for i in range(len(t)):
        error1[i]=np.linalg.norm((store[:,i]),2)
        error2[i]=np.linalg.norm((store2[:,i]),2)
        error3[i]=np.linalg.norm((store3[:,i]),2)
    print(error2)    
    error1=error1/error2
    error2=error2/error3
    plt.plot(t,np.log2(error1),color="red",label="Medium-Low/High-Medium")
    plt.plot(t,np.log2(error2),color="Blue",label="Higher-High/High-Medium")
    plt.ylim(0,4)
    plt.xlabel("Time")
    plt.ylabel("Norm Convergence Factor")
    plt.title("Norm Self Convergence")
    plt.legend()
    plt.show()
# Run the simulation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_wave(x, t, phi, interval=1000, frame_step=1, ylim=(-0.00002, 0.00002)):
    """
    Animates the evolution of the wave phi(x, t) over time.

    Parameters:
        x (1D np.ndarray): Spatial grid
        t (1D np.ndarray): Time grid
        phi (2D np.ndarray): phi[i, j] = phi(x_i, t_j)
        interval (int): Milliseconds between frames
        frame_step (int): Step size between animation frames
        ylim (tuple): Y-axis limits for the plot
    """
    fig, ax = plt.subplots()
    line_real, = ax.plot(x, np.real(phi[:, 0]), color="blue", label="Re(phi)")
    line_imag, = ax.plot(x, np.imag(phi[:, 0]), color="red", label="Im(phi)")
    
    ax.set_xlim(x[0], x[-1])
    #ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("phi")
    ax.set_title("Wave Equation Solution")
    ax.legend()

    def update(frame):
        line_real.set_ydata(np.real(phi[:, frame]))
        line_imag.set_ydata(np.imag(phi[:, frame]))
        ax.set_title(f"Spherical GR with xtra at t = {t[frame]:.2f} (RK2 Time Integration)")
        return line_real, line_imag

    ani = FuncAnimation(fig, update, interval=interval, frames=range(0, 10, frame_step), repeat_delay=10000)
    plt.show()
    # Optionally save:
    ani.save("going_toinfty.mp4")



animate_wave(*wave_solver(dx=4.0, dt=0.1))

#compute_pointwise_self(dx=4.0, dt=0.4)
#compute_norm_self(dx=4.0,dt=0.4)
