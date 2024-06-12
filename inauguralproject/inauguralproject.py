from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        par = self.par
        return x1A**par.alpha * x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        par = self.par
        return x1B**par.beta * x2B**(1-par.beta)

    def demand_A(self,p1):
        par = self.par
        x1A = par.alpha * ((p1*par.w1A + par.w2A)/p1)
        x2A = (1-par.alpha) * (p1*par.w1A + par.w2A)
        return x1A,x2A

    def demand_B(self,p1):
        par = self.par
        x1B = par.beta * ((p1*par.w1B + par.w2B)/p1)
        x2B = (1-par.beta) * (p1*par.w1B + par.w2B)
        return x1B,x2B
    
    def plot_edgeworth_box(self):
        par = self.par
        N = 75
        x1A_grid = np.linspace(0, 1, N)
        x2A_grid = np.linspace(0, 1, N)
        
        X1A, X2A = np.meshgrid(x1A_grid, x2A_grid)
        UA = self.utility_A(X1A, X2A)
        UB = self.utility_B(1-X1A, 1-X2A)
        
        UA_endowment = self.utility_A(par.w1A, par.w2A)
        UB_endowment = self.utility_B(par.w1B, par.w2B)
        
        plt.figure(figsize=(10, 10))
        
        plt.contour(X1A, X2A, UA, levels=[UA_endowment], colors='blue', linestyles='dotted')
        plt.contour(X1A, X2A, UB, levels=[UB_endowment], colors='red', linestyles='dotted')
        
        plt.fill_between(X1A.flatten(), X2A.flatten(), where=(UA >= UA_endowment) & (UB >= UB_endowment), color='green', alpha=0.3)
        
        plt.plot(par.w1A, par.w2A, 'bo', label='Endowment A')
        plt.plot(par.w1B, par.w2B, 'ro', label='Endowment B')
        
        plt.xlabel('Good 1')
        plt.ylabel('Good 2')
        plt.title('Edgeworth Box')
        plt.legend()
        plt.grid()
        plt.show()