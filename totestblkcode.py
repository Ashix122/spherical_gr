import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

from critical_collapse_prq import wave_solver,animate_wave



def binarysearch(a,b,tol=1e-13):
    "Binary Search Algorithm"
    while(np.abs(a-b)>tol):
        
        mid=(a+b)/2
        chk=blkholbm(mid)
        if (chk):
            b=mid
        elif( not chk):
            a=mid
        print(mid)
    return mid

def blkhol1(amp):
    _,_,phi,a,alpha=wave_solver(dx=4.0,dt=0.1,p=amp)
    if((np.isnan(phi[0,:]).any()) and (np.isnan(a).any())):
        return True
    else:
        return False

def blkhol2(amp):
    _,_,phi,a,alpha=wave_solver(dx=1.0,dt=0.1,p=amp)
    if((np.isnan(phi).any() or (np.abs(phi[0,:])>100*amp).any()) and (np.isnan(a).any())):
        return True
    else:
        return False
    
def blkholbm(amp):
    _,_,phi,a,alpha=wave_solver(dx=1.0,dt=0.1,p=amp)
    if (np.min(alpha)<1e-5):
        return True
    elif(np.linalg.norm(alpha-np.ones_like(alpha))<1e-5):
        return False


p=0.7
#p=0.32816722571
#p=0.328167225723
#p=0.3281672256
#p=0.001 
print(blkholbm(p))
#print("Final value=",np.round(binarysearch(0.001,1),decimals=10))
#animate_wave(*wave_solver(dx=10.0, dt=0.05,p=p)[:3],p=p)
x,t,phi,a,alpha=wave_solver(dx=1.0, dt=0.1,p=p)
print(alpha)
"""
x,t,phi,_,_=wave_solver(dx=1.0, dt=0.1,p=p)
plt.plot(t[:int(200/0.1)],phi[0,:int(200/0.1)])
plt.xlabel("t")
plt.ylabel("phi")
plt.title(f"Phi at origin with p={p:.10f}")
plt.savefig("BG-Subcritical.png")
plt.show()
"""
    

