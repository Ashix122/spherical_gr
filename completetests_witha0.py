import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wave_solver(dx, dt, T=1000):
    x = np.arange(0, 400+dx, dx)
    t = np.arange(0, T+dt, dt)
    nx, nt = len(x), len(t)

    if dt > dx:
        raise ValueError("CFL condition violated: Decrease dt or increase dx.")
    
    phi = np.zeros((nx, nt), dtype=complex)
    pi = np.zeros((nx, nt), dtype=complex)
    
    phi[:, 0] = 0.001 * np.exp(-0.5 * (x / 10)**2)
    pi[:, 0] = np.zeros_like(phi[:, 0])

    def laplacian(phi):
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

    def delr(v):
        derivative = np.zeros_like(v)
        for index in range(1, nx - 1):
            derivative[index] = (v[index + 1] - v[index - 1]) / (2 * dx)
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
    
    def koterm(v, dx=dx):
        term = np.zeros_like(v)
        for index in range(2, nx - 2):  # Avoid boundary regions
            term[index] = (v[index+2] - 4*v[index+1] + 6*v[index] - 4*v[index-1] + v[index-2]) / dx
        term[:2]=term[2]
        term[-2:]=term[-3]
        return -1*term * (0.02 / 16)

    def rhs(pi, phi,a,alpha):

        dpidt = laplacian(phi) * (alpha / a)**2 + (delr(alpha) * a - delr(a) * alpha) * delr(phi) / (a**2)
        dpidt[-1]= 1*(-3*pi[-1]+4*pi[-2]-pi[-3])/(2*dx)
        dpidt=dpidt+koterm(pi)
        dphidt = pi * (alpha / a)+koterm(phi)
        return np.array([dpidt, dphidt])

    # RK2 Time Integration
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
    store2 =4 * (phi_num_matched_vfine - phi_num_matched)
    store3 = 16 * (phi_num_matched_vvfine - phi_num_matched_vfine)
    
    point_plot(store, store2, store3, x, t,"self")

def point_plot(store, store2, store3, x,t,typer):
    fig, ax = plt.subplots()
    if typer == "self":
        a="Medium-low"
        b="4*(high-medium)"
        c="16*(higher-high)"
    elif typer == "exact":
        a="low-exact"
        b="4*Medium-exact"
        c="16*High-exact"
    line_real, = ax.plot(x[:], (store[:, 0]), color="blue", label=a)
    line_imag, = ax.plot(x[:], (store2[:, 0]), color="red", label=b)
    line3, = ax.plot(x[:], (store3[:, 0]), color="green", label=c)
    ax.set_xlim(0, 400)
    ax.set_ylim(-0.00025,0.00025)
    #ax.set_ylim(np.min([store, store2, store3]), np.max([store, store2, store3]))
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
    
    ani = FuncAnimation(fig, update, interval=50, frames=len(t), repeat_delay=10000)
    plt.show()
    ani.save("pointwise_self_a0.mp4")

def exact_solution(dx, dt,T=10):
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, T + dt, dt)
    nx, nt = len(x), len(t)
    phi = np.zeros((nx, nt), dtype=complex)

    for i in range(nx):
        for j in range(nt):
            phi[i,j]=np.sin(2 * np.pi * (x[i] - t[j]))

    return phi

def compute_norm_exact(dx, dt):
    
    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)
    store=phi_num-exact_solution(dx,dt)
    store2=(phi_num_fine-exact_solution(dx/2,dt/2))[:,::2]
    store3=(phi_num_vfine-exact_solution(dx/4,dt/4))[:,::4]
    store4=(phi_num_vvfine-exact_solution(dx/8,dt/8))[:,::8]
    error1=np.zeros_like(t)
    error2=np.zeros_like(t)
    error3=np.zeros_like(t)
    error4=np.zeros_like(t)
    for i in range(len(t)):
        error1[i]=np.linalg.norm((store[:,i])*dx**0.5,2)
        error2[i]=np.linalg.norm((store2[:,i])*(dx/2)**0.5,2)
        error3[i]=np.linalg.norm((store3[:,i])*(dx/4)**0.5,2)
        error4[i]=np.linalg.norm((store4[:,i])*(dx/8)**0.5,2)
        
    plt.plot(t,np.log2(error1/error2),color="red",label="(Medium-Exact)/(Low-Exact)")
    plt.plot(t,np.log2(error2/error3),color="blue",label="(High-Exact)/(Medium-Exact)")
    plt.plot(t,np.log2(error3/error4),color="green",label="(Higher-Exact)/(High-Exact)")
    plt.xlabel("Time")
    plt.ylim(0,10)
    plt.ylabel("Norm Convergence Factor")
    plt.title("Norm Exact Convergence")
    plt.legend()
    plt.show()
    plt.show()

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
    plt.savefig("NormSelf_changed.png")
def compute_pointwise_exact(dx,dt):   

    x, t, phi_num = wave_solver(dx, dt)
    x_fine, t_fine, phi_num_fine = wave_solver(dx/2, dt/2)
    x_vfine, t_vfine, phi_num_vfine = wave_solver(dx/4, dt/4)
    x_vvfine, t_vvfine, phi_num_vvfine = wave_solver(dx/8, dt/8)

    
    
    store = phi_num - exact_solution(dx,dt)
    store2 = 4 * (phi_num_fine - exact_solution(dx/2,dt/2))
    store3 = 16 * (phi_num_vfine - exact_solution(dx/4,dt/4))
    store = store
    store2 = store2[::2, ::2]
    store3 = store3[::4, ::4]
    
    point_plot(store, store2, store3, x, t,"exact")


compute_pointwise_self(dx=5.0, dt=0.5)   
#compute_norm_self(dx=4.0,dt=0.4)
#compute_norm_exact(1,0.1)
#compute_pointwise_exact(0.01,0.01)

