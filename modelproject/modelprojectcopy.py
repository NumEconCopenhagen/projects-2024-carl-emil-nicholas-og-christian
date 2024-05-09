from scipy import optimize
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

class Solowclass:
    def __init__(self):
        """
        Initializes the class.
        """
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

    def parameter_values(self):
        """
        Defines parameter values for both the baseline model and the extended model.
        """ 
        # Baseline Solow model parameters
        self.par.alpha = 0.2  # Output elasticity of capital
        self.par.s = 0.3      # Savings rate
        self.par.n = 0.01     # Population growth rate
        self.par.g = 0.027    # Technological growth rate
        self.par.delta = 0.05 # Depreciation rate

        # Extended Solow model parameters
        self.par.beta = 0.6  # Output elasticity of labor
        self.par.eps = 0.2   # Output elasticity of the exhaustible resource
        self.par.sE = 0.005  # Fraction of the resource consumed each period
        self.par.phi = 0.5   # Climate damage parameter

    def solve_ss_z_par(self, zss):
        """
        Solves for the steady state of the capital-output ratio (z) in the extended model.
        """
        objective = lambda zss: zss - (1/(1 - self.par.sE))**(self.par.eps + self.par.phi) * \
            (1/((1 + self.par.n) * (1 + self.par.g)))**self.par.beta * \
            (self.par.s + (1 - self.par.delta) * zss)**(1 - self.par.alpha) * zss**self.par.alpha
        result = optimize.root_scalar(objective, bracket=[0.1, 100], method='brentq')
        print(f'The steady state for z in the Solow model with an exhaustible resource and climate change is {result.root}')

    def solve_ss_k_par(self, kss):
        """
        Solves for the steady state of capital per effective worker (k) in the standard model.
        """
        production_function = lambda k: k**self.par.alpha
        objective = lambda kss: kss - (self.par.s * production_function(kss) + (1 - self.par.delta) * kss) / \
            ((1 + self.par.g) * (1 + self.par.n))
        result = optimize.root_scalar(objective, bracket=[0.1, 100], method='brentq')
        print(f'The steady state for k in the standard Solow model is {result.root}')

    def simulate(self, T, k0, l0, a0, r0):
        """
        Simulates the model over T periods starting from initial values.
        """
        # Initialize arrays for simulation variables
        self.sim.k = np.empty(T)
        self.sim.z = np.empty(T)
        self.sim.y = np.empty(T)
        self.sim.a = np.empty(T)
        self.sim.e = np.empty(T)
        self.sim.d = np.empty(T)
        self.sim.r = np.empty(T)
        self.sim.l = np.empty(T)

        # Set initial values
        self.sim.k[0] = k0
        self.sim.l[0] = l0
        self.sim.a[0] = a0
        self.sim.r[0] = r0
        self.sim.e[0] = self.par.sE * self.sim.r[0]
        self.sim.d[0] = 1 - (self.sim.r[0] / self.sim.r[0])**self.par.phi
        self.sim.y[0] = (1 - self.sim.d[0]) * self.sim.k[0]**self.par.alpha * (self.sim.a[0] * self.sim.l[0])**self.par.beta * self.sim.e[0]**(1 - self.par.alpha - self.par.beta)
        self.sim.z[0] = self.sim.k[0] / self.sim.y[0]

        # Time loop for simulation
        for t in range(1, T):
            self.sim.l[t] = (1 + self.par.n) * self.sim.l[t-1]
            self.sim.a[t] = (1 + self.par.g) * self.sim.a[t-1]
            self.sim.r[t] = self.sim.r[t-1] - self.sim.e[t-1]
            self.sim.e[t] = self.par.sE * self.sim.r[t]
            self.sim.d[t] = 1 - (self.sim.r[t] / self.sim.r[0])**self.par.phi
            self.sim.k[t] = (1 - self.par.delta) * self.sim.k[t-1] + self.par.s * self.sim.y[t-1]
            self.sim.y[t] = (1 - self.sim.d[t]) * self.sim.k[t]**self.par.alpha * (self.sim.a[t] * self.sim.l[t])**self.par.beta * self.sim.e[t]**(1 - self.par.alpha - self.par.beta)
            self.sim.z[t] = self.sim.k[t] / self.sim.y[t]

        # Plot the results
        plt.figure()
        plt.plot(self.sim.z, label='Capital-output ratio over time')
        plt.axhline(y=4.09, color='blue', linestyle='--', label='Target z value')
        plt.legend()
        plt.show()

        return self.sim
