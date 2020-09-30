from __future__ import division
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from numba import jit
from numpy.random import rand


Nx=Ny=64
N_measure=1500
N_therma=1000
J= 1
h= 0
beta=0.47
beta_crit=np.log(1+np.sqrt(2))/2
J_tilde=J*beta
h_tilde=h*beta

class Ising():
    ###SW-algorithm was very nicely written by my dear colleague Mr. Mark Petry
    ### and then included into my class Ising
    def Swendsenwang(self, spin, beta):
        Nx=spin.shape[0]
        Ny=spin.shape[1]
        bond=self.bonds(spin)
        cluster=self.HK(bond)
        cluster= self.clusterrecycling(cluster, Nx, Ny)
        spin= self.clusterflip(spin,cluster)
        return spin

    def bonds(self, spin):
        Nx=spin.shape[0]
        Ny=spin.shape[1]
        bond=np.zeros((Nx,Ny,2),dtype=np.int64)
        rand_sp=np.random.rand(Nx,Ny,2)
        growprob=1-np.exp(-2)
        for j in range(0,Nx):
            j_plu= (j+1)%Nx
            for k in range(0,Ny):
                k_plu=(k+1)%Ny
                bond[j,k,0]=spin[j,k]*spin[j_plu,k]>0.0 and (rand_sp[j,k,0]<growprob)
                bond[j,k,1] = spin[j,k]*spin[j,k_plu]>0.0 and (rand_sp[j,k,1]<growprob)
        return bond
    
    def HK(self, bond):
        Nx=bond.shape[0]
        Ny=bond.shape[1]
        cluster=- np.ones((Nx*Ny),dtype=np.int64) #evtl hier int32 siehe oben np.int_
        for x in range(0,Nx):
            x_plu = ((x + 1) % Nx)
            for y in range(0,Ny):
                y_plu = ((y+1)%Ny)
                n=Ny*x+y
                if bond[x,y,1]==1: #right
                    m=(Ny*x+y_plu)
                    root_n=self.findroot(n,cluster)
                    root_m=self.findroot(m,cluster)
                    if root_n!=root_m:
                        entry_n=cluster[root_n]
                        entry_m=cluster[root_m]
                        if entry_m<entry_n:
                            cluster[root_m]= entry_n+entry_m
                            cluster[root_n]= root_m
                        else:
                            cluster[root_n]=entry_n+entry_m
                            cluster[root_m]=root_n
                if bond[x,y,0]==1: #below
                    m=(Ny*x_plu+y)
                    root_n = self.findroot(n, cluster)
                    root_m = self.findroot(m, cluster)
                    if root_n != root_m:
                        entry_n = cluster[root_n]
                        entry_m = cluster[root_m]
                        if entry_m < entry_n:
                            cluster[root_m] = entry_n + entry_m
                            cluster[root_n] = root_m
                        else:
                            cluster[root_n] = entry_n + entry_m
                            cluster[root_m] = root_n
        return cluster
    
    def findroot(self, x,cluster):
        y=x
        while cluster[y]>=0:
            y=cluster[y]
        root=y
        return root    
    
    def clusterrecycling(self, cluster,Nx,Ny):
        for n in range(0,Nx*Ny):
            if cluster[n]>=0:
                cluster[n]= self.findroot(n,cluster)
        for n in range(0,Nx*Ny):
            if cluster[n]<0:
                cluster[n]=n
        return cluster
    
    def clusterflip(self, spin,cluster):
        Nx = spin.shape[0]
        Ny = spin.shape[1]
        labelsarray = np.ones(max(cluster)+1,dtype=np.int64)
        for j in cluster:
            if np.random.random()<0.5:
                labelsarray[j]=-np.float64(1)
        for n in range(0,Nx*Ny):
            x=n//Ny
            y=n%Ny
            spin[x,y]*=labelsarray[cluster[n]]
        return spin
    
    
    def HeatbathRandom(self, spin, beta):
        Nx = spin.shape[0]
        Ny = spin.shape[1]
        z_rand = np.random.randint(0,Nx,Nx)
        s_rand = np.random.randint(0,Ny,Ny)
        for j in z_rand:
            j_plu = (j + 1) % Nx
            j_min = (j - 1) % Nx
            for k in s_rand:
                NNSumm = (spin[j, (k + 1) % Ny] + spin[j_plu, k] + spin[j_min, k] + spin[j, (k - 1) % Ny])
                pos_prob = np.exp(J_tilde * NNSumm) / (np.exp(-J_tilde * NNSumm) + np.exp(J_tilde * NNSumm))
                if np.random.random() < pos_prob:
                    spin[j, k] = 1.0
                else:
                    spin[j, k] = -1.0
        return spin
    
    def mcMetroRandom(self, spinconfig,n , beta):
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
    
    
    def configPlot(self, f, spin, i, n, n_):
        ''' This modules plts the configuration once passed to it along with time etc '''
        X, Y = np.meshgrid(range(n), range(n))
        sp =  f.add_subplot(4, 4, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.imshow(spin, cmap='CMRmap_r');
        plt.title('MC-Steps=%d'%i); plt.axis('tight')
        #plt.savefig('SWEqui64.pdf_beta=0.47',format='pdf')
        #plt.savefig('MetroEqui64_MCSTEPS_10E5.pdf',format='pdf')
    plt.show()
    
    def simulate(self):   
        ''' This module simulates the Ising model'''
        n, beta, temp, N , Nx, Ny     = 64,  .4407 , .4407   , 64,64,64   # Initialice the lattice
        spin = 2*np.random.randint(2, size=(N,N))-1
        f = plt.figure(figsize=(15, 15), dpi=80);    
        self.configPlot(f, spin, 0, N, 1);
        
        msrmnt = 20001
        for i in range(msrmnt):
            
            self.Swendsenwang(spin, 0.5)
            #self.HeatbathRandom(spin, 0.5)
            #self.mcMetroRandom(spin, 1.0/2.0)
            if i == 1:       self.configPlot(f, spin, i, N, 2);
            if i == 4:       self.configPlot(f, spin, i, N, 3);
            if i == 8:      self.configPlot(f, spin, i, N, 4);
            if i == 16:     self.configPlot(f, spin, i, N, 5);
            if i == 50:    self.configPlot(f, spin, i, N, 6);
            if i == 100:    self.configPlot(f, spin, i, N, 7);
            if i == 200:    self.configPlot(f, spin, i, N, 8);
            if i == 300:    self.configPlot(f, spin, i, N, 9);
            if i == 500:    self.configPlot(f, spin, i, N, 10);
            if i == 1000:    self.configPlot(f, spin, i, N, 11);
            if i == 1500:    self.configPlot(f, spin, i, N, 12);
            if i == 2000: self.configPlot(f, spin, i, N, 13);
            if i == 5000:   self.configPlot(f, spin, i, N, 14);
            if i == 10000:   self.configPlot(f, spin, i, N, 15);
            if i == 20000:   self.configPlot(f, spin, i, N, 16);
            
####output (depending from the algorithm)
sim = Ising()
sim.simulate()
