class Ising():
    ''' Simulating the Ising model '''
    
    ## monte carlo moves
    def mcmove(self, config, N, beta):
        ''' This is to execute the monte carlo moves using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
        for i in range(N):
            for j in range(N):
                    a = np.random.randint(0, N)
                    b = np.random.randint(0, N)
                    spinlattice =  config[a, b] #initvalue of spinLattice
                
                    # Periodic Boundary Condition
                    neighbours = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                    
                    # change in energy:
                    Delta_E=2*spinlattice*neighbours
                    
                    # using acceptance test:
                    if Delta_E<0:
                      spinlattice *=-1
                    elif rand()< np.exp(-Delta_E*beta):
                      spinlattice *=-1
                      
                    # satisfing the detailed balance condition & ensuring a final equilibrium state. And new config is:
                    config[a, b] = spinlattice
        return config

