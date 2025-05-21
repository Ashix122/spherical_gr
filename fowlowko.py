import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

from critical_collapse_prq import wave_solver,animate_wave



def binarysearch(a,b,tol=1e-13):
    "Binary Search Algorithm"
    while(np.abs(a-b)>tol):
        
        mid=(a+b)/2
        chk=blkhol2(mid)
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
    _, _, phi, a, alpha = wave_solver(dx=1.0, dt=0.1, p=amp)
    if np.isnan(alpha).any():
        return True  # Treat NaNs as indicative of collapse
    elif np.min(alpha) < 1e-3:
        return True  # Lapse collapsed: black hole
    elif np.linalg.norm(alpha - 1.0) < 1e-5:
        return False  # Flat spacetime: dispersal
    else:
        # Intermediate state — not clearly black hole or dispersal
        return False  # or raise an exception if you want strict behavior

def plotphi0(p):
    x,t,phi,_,_=wave_solver(dx=1.0, dt=0.1,p=p)
    plt.plot(t[:int(600/0.1)],phi[0,:int(600/0.1)])
    plt.xlabel("t")
    plt.ylabel("phi")
    plt.title(f"Phi at origin with p={p:.16f}")
    plt.savefig("BG_withlowKO.png")
    plt.show()
    

def digitsearch(a,g):
    for i in range(1,11):
      bmin=a+(i-1)*(10**(-g))
      b=a+(i)*(10**(-g))
      if g==17:
        return a
      if(blkholbm(b)):
          return digitsearch(bmin,g+1)
      else:
          print(f"not {b:.16f}")
p=0.3377534636
#p=0.3407818470201690
#p=0.3407818470
#print("Final value=",np.round(binarysearch(0.001,1),decimals=10))
animate_wave(*wave_solver(dx=1.0, dt=0.1,p=p)[:3],p=p)
#plotphi0(p)
#fp=digitsearch(p,11)
#print(f"Final value={fp:.16f}")
#plotphi0(fp)