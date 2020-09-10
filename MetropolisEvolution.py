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
    
    def simulate(self):   
        ''' This module simulates the Ising model'''
        n, temp, N     = 64, .4   , 64     # Initialice the lattice
        spinconfig = 2*np.random.randint(2, size=(N,N))-1
        f = plt.figure(figsize=(15, 15), dpi=80);    
        self.configPlot(f, spinconfig, 0, N, 1);
        
        msrmnt = 10001
        for i in range(msrmnt):
            self.mcmove(spinconfig, N, 1.0/temp)
            if i == 1:       self.configPlot(f, spinconfig, i, N, 2);
            if i == 4:       self.configPlot(f, spinconfig, i, N, 3);
            if i == 32:      self.configPlot(f, spinconfig, i, N, 4);
            if i == 100:     self.configPlot(f, spinconfig, i, N, 5);
            if i == 1000:    self.configPlot(f, spinconfig, i, N, 6);
            if i == 2000:    self.configPlot(f, spinconfig, i, n, 7);
            if i == 5000:    self.configPlot(f, spinconfig, i, n, 8);
            if i == 10000:    self.configPlot(f, spinconfig, i, n, 9);
                
    def configPlot(self, f, spinconfig, i, n, n_):
        ''' This module plts the configuration once passed to it along with time etc '''
        X, Y = np.meshgrid(range(n), range(n))
        sp =  f.add_subplot(3, 3, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, spinconfig, cmap=plt.cm.RdBu);
        plt.title('Time=%d'%i); plt.axis('tight')
        plt.savefig('metroEqui.pdf',format='pdf')
