import numpy as np
import matplotlib.pyplot as plt

class SolowModel:
    def __init__(self, alpha=0.33, s=0.20, delta=0.05, n=0.02, g=0.02, K0=100, L0=100, T=100):
        self.alpha = alpha 
        self.s = s
        self.delta = delta
        self.n = n
        self.g = g
        self.K0 = K0
        self.L0 = L0
        self.T = T
        self.K = np.zeros(T)
        self.L = np.zeros(T)
        self.Y_base = np.zeros(T)

    def simulate(self):
        self.K[0] = self.K0
        self.L[0] = self.L0

        for t in range(self.T-1):
            self.Y_base[t] = self.K[t]**self.alpha * self.L[t]**(1-self.alpha)
            self.K[t+1] = self.s * self.Y_base[t] + (1 - self.delta) * self.K[t]
            self.L[t+1] = (1 + self.n) * self.L[t]
        
        self.Y_base[self.T-1] = self.K[self.T-1]**self.alpha * self.L[self.T-1]**(1-self.alpha)

    def plot_results(self):
        plt.figure(figsize=(10, 8))
        plt.plot(self.K, label='Capital')
        plt.plot(self.L, label='Labor')
        plt.plot(self.Y_base, label='Output')
        plt.title('Figure 1: Solow Model Simulation')
        plt.xlabel('Time')
        plt.ylabel('Levels')
        plt.legend()
        plt.grid(True)
        plt.show()
