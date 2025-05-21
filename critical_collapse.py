import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

from critical_collapse_prq import wave_solver



def binarysearch(a,b,tol=1e-8):
    "Binary Search Algorithm"
    while(np.abs(a-b)>tol):
        
        mid=(a+b)/2
        if (blkhol1(mid)):
            b=mid
        else:
            a=mid
        print (mid)
    return mid

def blkhol1(amp):
    _,_,phi,a,alpha=wave_solver(dx=4.0,dt=0.1,p=amp)
    if((np.isnan(phi[0,:]).any()) and (np.isnan(a).any())):
        return True
    else:
        return False

#print(blkhol1(1))
binarysearch(1,0.001)

    

