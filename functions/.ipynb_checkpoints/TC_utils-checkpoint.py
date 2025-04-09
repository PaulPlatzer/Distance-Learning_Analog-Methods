## All codes written by Arthur Avenas and Paul Platzer.
# These codes are attached to the following publication:
# Paul Platzer, Arthur Avenas, Bertrand Chapron, Lucas Drumetz, Alexis Mouche, Pierre Tandeo, Léo Vinour, Distance Learning for Analog Methods. 2024. ⟨hal-04841334⟩

import numpy as np

def M(f, r, V):
    '''Definition of angular momentum.
    CAVEAT: R is in meters, V in m/s.
    If V is total wind speed, then take abs(f)'''
    return r * V + 0.5 * f * (r ** 2)


def Rmax_from_M(fcor, Mmax, Vmax):
    return (Vmax / fcor) * (np.sqrt(1 + (2 * fcor * Mmax) / (Vmax ** 2)) - 1)


def correct_vmx_ibt(vmx_ibt):
    return 0.6967 * vmx_ibt + 6.1992
# "Correct" Vmax from IBTrACS to a 1D Vmax


def Rmxa23(fcor, Vmax, R34):
    vmx  = correct_vmx_ibt(Vmax)
    
    # Compute the momentum ratio
    Mm_M34 = 0.531 * np.exp(
        -0.00214 * (vmx - 17.5)
        -0.00314 * (vmx - 17.5) * 0.5 * fcor * R34 * 1000
    )

    # Compute Mmax
    Mm_with_r34 = Mm_M34 * M(fcor, R34 * 1000, 17.5)

    # Compute Rmax_A23
    Rmax_A23 = Rmax_from_M(fcor, Mm_with_r34, vmx) / 1000
    
    return Rmax_A23