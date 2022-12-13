import numpy as np
from collections import deque
from numba import jit

def Flipflop(spin,J_tilde,h_tilde):
    Nx=spin.shape[0]
    Ny=spin.shape[1]
    acc=0
    e=Energy(spin, J_tilde,h_tilde)
    spin_new=-spin
    e_new=Energy(spin_new,J_tilde,h_tilde)
    if rand()<np.exp(-Nx*Ny*(e_new-e)):
        spin=spin_new
        acc+=1
    return spin,acc

@jit(nopython=True)
def Energy(spin, J_tilde, h_tilde):
    Nx = spin.shape[0]  # Anzahl Zeilen
    Ny = spin.shape[1]  # Anzahl Spalten
    energy = 0.0
    for j in range(0, Nx):
        j_plu = (j + 1) % Nx
        for k in range(0, Ny):
            energy += -J_tilde * (spin[j, (k + 1) % Ny] + spin[j_plu, k]) * spin[j, k] - h_tilde * spin[j, k]
    return energy / float(Nx * Ny)

def Matrix(beta,beta_crit,Nx,Ny):
    if beta < beta_crit:
        spin = np.random.rand(Nx, Ny)
        spin_pos = spin < 0.5
        spin_neg = spin >= 0.5
        spin[spin_pos] = 1.0
        spin[spin_neg] = -1.0
    else:
        spin = np.ones([Nx,Ny])
    return spin


@jit(nopython=True)
def Swendsenwang(spin,J_tilde):
    Nx=spin.shape[0]
    Ny=spin.shape[1]
    bond=bonds(spin,J_tilde)
    cluster=HK(bond)
    cluster=clusterrecycling(cluster, Nx, Ny)
    spin=clusterflip(spin,cluster)
    return spin
@jit(nopython=True)
def bonds(spin,J_tilde):
    Nx=spin.shape[0]
    Ny=spin.shape[1]
    bond=np.zeros((Nx,Ny,2),dtype=np.int64)
    rand_sp=np.random.rand(Nx,Ny,2)
    growprob=1-np.exp(-2*J_tilde)
    for j in range(0,Nx):
        j_plu= (j+1)%Nx
        for k in range(0,Ny):
            k_plu=(k+1)%Ny
            bond[j,k,0]=spin[j,k]*spin[j_plu,k]>0.0 and (rand_sp[j,k,0]<growprob)
            bond[j,k,1] = spin[j,k]*spin[j,k_plu]>0.0 and (rand_sp[j,k,1]<growprob)
    return bond


@jit(nopython=True)
def clusterflip(spin,cluster):
    Nx = spin.shape[0]
    Ny = spin.shape[1]
    labelsarray = np.ones(max(cluster)+1,dtype=np.int64)#hier evtl int anstatt float, da cluster auch int
    for j in cluster:
        if np.random.random()<0.5:
            labelsarray[j]=-np.float64(1)
    for n in range(0,Nx*Ny):
        x=n//Ny
        y=n%Ny
        spin[x,y]*=labelsarray[cluster[n]]
    return spin


@jit(nopython=True)
def HK(bond):
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
                root_n=findroot(n,cluster)
                root_m=findroot(m,cluster)
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
                root_n = findroot(n, cluster)
                root_m = findroot(m, cluster)
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

@jit(nopython=True)
def findroot(x,cluster):
    y=x
    while cluster[y]>=0:
        y=cluster[y]
    root=y
    return root

@jit(nopython=True)
def clusterrecycling(cluster,Nx,Ny):
    for n in range(0,Nx*Ny):
        if cluster[n]>=0:
            cluster[n]=findroot(n,cluster)
    for n in range(0,Nx*Ny):
        if cluster[n]<0:
            cluster[n]=n
    return cluster

def correlator(spin):
    Nx = spin.shape[0]
    Ny = spin.shape[1]
    correlator = np.zeros([(Nx//2)+1])
    timeslice=np.sum(spin,axis=1)/Ny
    for x in range(0, Nx):
        for x_p in range(0, Nx):
            d=min(abs(x-x_p),Nx-abs(x-x_p))
            correlator[d]+=timeslice[x]*timeslice[x_p]
    correlator/=Nx
    correlator[0]*=2;correlator[Nx//2]*=2 ##ersetzen durch corr 1 bis nx/2-1 *0.5
    correlator/=2
    return correlator

from mpl_toolkits.axes_grid1 import make_axes_locatable
def scatter_hist_plt(x,gauss_fit=True,ylabel=''):
    mu=np.mean(x)
    sigma=np.std(x)
    bins = 30
    fig, ax = plt.subplots(figsize=(18, 10))

    # the scatter plot:
    ax.plot(x,'.')
    ax.set_ylabel(ylabel,  fontsize=20)
    ax.set_xlabel(r'MC-Steps', fontsize=20)
    divider = make_axes_locatable(ax)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    axHisty.yaxis.set_tick_params(labelleft=False)
    axHisty.set_xticks([0.25])

    # histogram
    n, bins, patches=axHisty.hist(x, bins=bins, orientation='horizontal',
                                  density=True)
    if gauss_fit:
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        axHisty.plot(y,bins, '-')
    #plt.savefig('E_150000MC_SW128_Avg=0.pdf',format='pdf')
    plt.show()
    
    
###Initialisation###
spin= Matrix(beta,beta_crit,Nx,Ny)
###Thermalisierung###
for i in range(N_therma):
    for j in range(N_Sepa):
        spin,_,_=Swendsenwang(spin,J_tilde)
        
###SwendsenWang###
for i in range(N_measure):
    spin,_=Flipflop(spin,J_tilde,h_tilde)
    for j in range(N_Sepa):
        spin=Swendsenwang(spin,J_tilde)
    #energy= Energy(spin,J_tilde,h_tilde);mag = magn(spin);magn_list.append(mag); energy_list.append(energy);size_list.append(meansize);invsize_list.append(invmeansize)
    #correlation=correlator(spin)
    #corr_array=np.vstack((corr_array,correlation))
    #print(int(i),energy,mag)
    #measure_list.append([i,energy_list[i],magn_list[i],size_list[i],invsize_list[i]])
#np.savetxt('measureSW128beta=0.4407_150000MCSteps_.txt',measure_list[0:],fmt='%1.5f',delimiter='\t')
#np.savetxt('corrSW64_MC1500.txt',corr_array,fmt='%1.5f',delimiter='\t')


####output
data = np.loadtxt('measureSWbeta=0.4407_1500MCSteps_.txt').T
e = data [1]
m= data[2]

scatter_hist_plt(e,ylabel=r'$E$',gauss_fit=True)
scatter_hist_plt(abs(m),ylabel=r'$\langle \vert M \vert  \rangle$', gauss_fit=False)
print(f"average of energy = {np.mean(e):.3f} ± {np.std(e):.3f}")
print(f"average of magnetization = {np.mean(abs(m)):.5f} ±" +f"{np.std(abs(m)):.5f}")
