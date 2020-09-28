# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:36:55 2020

@author: SePa
"""

from __future__ import division
from numba import jit
import numpy as np


N = 64
nt = 1000 #  number of temperature points
eqSteps = 1000    #  number of MC sweeps for equilibration (default:1024)
mcSteps = 1500       #  number of MC sweeps for calculation (default:1024)

T = np.linspace(1., 5., nt);
T_c = 2.269
beta_c=np.log(1+np.sqrt(2))/2
T_crit = 1.0 / beta_c

beta = np.linspace(0.35, 0.5, nt)
M=np.zeros(nt)
Chi=np.zeros(nt)
B_L = np.zeros(nt)
E = np.zeros(nt)
Energy = np.zeros(nt)
C = np.zeros(nt)
B = np.zeros(nt)


###functions
def magnetization(spinconfig):
    i = int
    sum = np.float64
    sum = 0.
    size = N*N
    for i in range(0,size):
        sum += spinconfig[i]
    return (2.0*sum - size) / size

def magn(spinconfig):
    magn = np.sum(spinconfig)  / (float(N*N))
    return magn


# initial rdm spin configuration:
def initialLattice(n):
    '''Create a nxn lattice with random spin configuration'''
    spinLattice = np.random.choice([1,-1], size=(n,n))
    return spinLattice

def coldinitialLattice(n):
    spinLattice = np.ones( (N,N), dtype=np.int64  )
    return spinLattice


def calcEnergy(spinconfig):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(spinconfig)):
        for j in range(len(spinconfig)):
            S = spinconfig[i,j]
            nb = spinconfig[(i+1)%N, j] + spinconfig[i,(j+1)%N] + spinconfig[(i-1)%N, j] + spinconfig[i,(j-1)%N]
            energy += -nb*S
    return energy/(float(N*N))

@jit()
def mcMetroRandom(spinconfig,n , beta):
    ''' This is to execute the monte carlo moves using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
    for i in range(n):
        for j in range(n):
            a=np.random.randint(0,n) # looping over i & j therefore use a & b
            b=np.random.randint(0,n)
            spinlattice=spinconfig[a,b]   # is initvalue of spinLattice
            
            # Periodic Boundary Condition
            neighbours=spinconfig[(a+1)%n, b] + spinconfig[a, (b+1)%n] + spinconfig[(a-1)%n, b] + spinconfig[a, (b-1)%n]
            
            # change in energy:
            Delta_E=2*spinlattice*neighbours
            
            # using acceptance test:
            if Delta_E<0:
                spinlattice=-1*spinlattice
            elif np.random.random()< np.exp(-Delta_E*beta):
                spinlattice=-1*spinlattice
            
            # anyway: satisfing the detailed balance condition, 
            # ensuring a final equilibrium state. And new config is:
            spinconfig[a,b]=spinlattice
    return spinconfig


#----------------------------------------------------------------------
#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
#@jit()
for t in range(nt):
    M0=0; M1=0; M2 = 0;
    E0=0; E1=0; E2 = 0;
    #spinconfig=init_config(N)
    
    ##cold start
    spinconfig = coldinitialLattice(N)
    
    beta = 1./T[t]
    beta_square = beta *beta
    #config = initialstate(N)
    iT=1.0/T[t]; iT2=iT*iT;
    
    for i in range(eqSteps):         # equilibrate
        mcMetroRandom(spinconfig, N, beta)
        #mcmove(config, iT)           # Monte Carlo moves

    for i in range(mcSteps):
        #spinconfig = Flipflop(spinconfig)  
        mcMetroRandom(spinconfig,N,beta)
        #mcmove(config, iT)           
        #Mag=calcMagnetisation(spinconfig)
        #Mag = calcMag(config)        # calculate the magnetisation
        Mag = abs(magn(spinconfig))
        chi = Mag * Mag
        Ene = calcEnergy(spinconfig)     # calculate the energy
        heat = Ene * Ene
    
        ### add all quantities
        #M0 += Mag  #magnetization
        M1 += chi  ## suscep
        M2 += chi * chi   ## for binder and also variance of suscep
        
        #### energystuff
        E0 += Ene ###energy
        E1 += heat ### for specific heat
        E2 += heat * heat #### for the variance of the specific heat
        
    # appending the values
    #E[t] = E0/(1.0 * mcSteps)
    Chi[t] = (M2/(1.0*mcSteps) - M1/(1.0 * mcSteps) * M1/(1.0 * mcSteps))*iT*N*N
    C[t] = (E2/(1.0*mcSteps) - E1/(1.0 * mcSteps) * E1/(1.0 * mcSteps))*iT2*N*N
    #M[t]=M0/(1.0 * mcSteps)
    #B_L[t] = 1.0 - (M2/(1.0*mcSteps))/(3.0 * (M1/(1.0*mcSteps)) * (M1/(1.0*mcSteps)))
