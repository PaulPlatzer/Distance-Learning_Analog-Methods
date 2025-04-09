### All codes written by Paul Platzer ###

### These codes are provided to reproduce figures from the following article:
### Paul Platzer, Arthur Avenas, Bertrand Chapron, Lucas Drumetz, Alexis Mouche, et al.. Distance Learning for Analog Methods. 2024. ⟨hal-04841334⟩
### https://hal.science/hal-04841334v1

### The Lorenz system was introduced in Lorenz, E. N. (1963). Deterministic nonperiodic flow. Journal of atmospheric sciences, 20(2), 130-141.

import numpy as np
from tqdm.notebook import tqdm

# Runge-Kutta 4 explicit
def RK4(yt,h,f):
    k1=h*f(yt)
    k2=h*f(yt+0.5*k1)
    k3=h*f(yt+0.5*k2)
    k4=h*f(yt+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return yt+dy

## Equations of the Lorenz (1963) system
def l63(Xt, sigma = 10, rho = 28, beta = 8/3):
    xdot=sigma*(Xt[1]-Xt[0])
    ydot=Xt[0]*(rho-Xt[2])-Xt[1]
    zdot=Xt[0]*Xt[1]-beta*Xt[2]
    return np.array([xdot,ydot,zdot])

## Integrate over one trajectory
def integrate_l63( X0 = np.array([1,1,1]) , dt = 0.01 , N = 10**5 , spin_up = 10**3 ):
    # Spin-up so that X0 belongs to the attractor
    for i in range(spin_up):
        X0 = RK4(X0, dt, l63)
        
    # Initialize
    traj = np.full( (N, 3) , np.nan )
    traj[0] = X0

    # Integrate
    for i in tqdm(range(N-1)):
        traj[i+1] = RK4(traj[i], dt, l63)

    return traj