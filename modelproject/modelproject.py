from scipy import optimize
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import sympy as sm

class Solowclass:
    def __init__(self):
        """
        initialises class
        """
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

    def paramtervalues(self):
        """
          Defining parameter values for both the baseline model and the extended model. 
        """ 

        par=self.par

        # ISN: JEG HAR INDSAT VÆRDIERNE FRA EKSAMENSSÆTTET
        # Baseline Solow model
        par.alpha=0.2
        par.s=0.3
        par.n=0.01 
        par.g=0.027
        par.delta=0.05 

        # Extended Solow model
        par.beta=0.6
        par.eps=0.2
        par.sE=0.005
        par.phi=0.5

    # ISN: KOPIERET ASH'S LØSNINGER MEN INDSAT PARAMETERVÆRDIER 

    def solve_ss_z_par(self, zss):

        par=self.par

        obj_zss = lambda zss: zss-(1/(1-par.sE))**(par.eps+par.phi)*(1/((1+par.n)*(1+par.g)))**par.beta*(par.s+(1-par.delta)*zss)**(1-par.alpha)*zss**par.alpha
        result = optimize.root_scalar(obj_zss,bracket=[0.1,100],method='brentq')

        print('The steady state for z in the Solow model with an exhaustable resource and climate change is',result.root)

    def solve_ss_k_par(self, kss):
        par=self.par

        f = lambda k: k**par.alpha
        obj_kss = lambda kss: kss - (par.s*f(kss) + (1-par.delta)*kss)/((1+par.g)*(1+par.n))
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        print('The steady state for k in the standard Solow model is',result.root)   
        
    
    def simulate(self,T,k0,l0,a0,r0):
        
        par=self.par
        sim=self.sim

        # Initialize arrays
        sim.k = np.empty(T)
        sim.z = np.empty(T)
        sim.y = np.empty(T)
        sim.a = np.empty(T)
        sim.e = np.empty(T)
        sim.d = np.empty(T)
        sim.r = np.empty(T)
        sim.l = np.empty(T)

        # Initial values
        sim.k[0] = k0
        sim.l[0] = l0
        sim.a[0] = a0
        sim.r[0] = r0
        sim.e[0] = par.sE*sim.r[0]
        sim.d[0] = 1-(sim.r[0]/sim.r[0])**par.phi
        sim.y[0] = (1-sim.d[0])*sim.k[0]**par.alpha*(sim.a[0]*sim.l[0])**par.beta*sim.e[0]**(1-par.alpha-par.beta)
        sim.z[0] = sim.k[0]/sim.y[0] 

        # Simulate
        for t in range(1,T):
            sim.l[t] = (1+par.n)*sim.l[t-1]
            sim.a[t] = (1+par.g)*sim.a[t-1] 
            sim.r[t] = sim.r[t-1] - sim.e[t-1]
            sim.e[t] = par.sE*sim.r[t]
            sim.d[t] = 1-(sim.r[t]/sim.r[0])**par.phi
            sim.k[t] = (1-par.delta)*sim.k[t-1]+par.s*sim.y[t-1]
            sim.y[t] = (1-sim.d[t])*sim.k[t]**par.alpha*(sim.a[t]*sim.l[t])**par.beta*sim.e[t]**(1-par.alpha-par.beta)
            sim.z[t] = sim.k[t]/sim.y[t]

        # plot
        fig = plt.figure()
        plt.plot(sim.z, label='z')
        plt.axhline(y=4.09, color='b', linestyle='--')
        plt.legend()
        plt.show()

        return sim

    # define simulation with widget for parameters
    def simulation_widget(self,alpha=0.2,beta=0.6,phi=0.5):
        par = self.par

        par.alpha= alpha
        par.beta = beta
        par.eps = 1-alpha-beta
        par.phi = phi

        # check for errors in parameters
        if par.alpha + par.beta  > 1:
            raise ValueError('alpha + beta must be less than 1')

        # simulate
        simulation = self.simulate(T=100,k0=1,l0=1,a0=1,r0=1)

        return simulation