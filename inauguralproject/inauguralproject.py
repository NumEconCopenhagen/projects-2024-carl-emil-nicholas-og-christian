# 1. Import relevant packages
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from scipy import optimize 
from scipy.optimize import minimize

# 2. Defines the class that are used in the project
class EdgeworthBoxClass:
    # a. Define the init function
    def __init__(self, alpha, beta, endowment_A, num_pairs=50):
        self.alpha = alpha
        self.beta = beta
        self.endowment_A = endowment_A
        self.endowment_B = [1 - e for e in endowment_A]

        # Set the number of allocations
        self.N = 75
        self.num_pairs = num_pairs
        self.pairs = None

        # 
        self.num_pairs = num_pairs

    # c. Define the utility and demand functions for the consumers
    
    # 1. Define the utility function for consumer A
    def u_A(self, x_A1, x_A2):
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    # 2. Define the utility function for consumer B
    def u_B(self, x_B1, x_B2):
        # i. Returns the value of the utility function for consumer B
        return x_B1**self.beta * x_B2**(1 - self.beta)

    # 3. Define the demand function for good 1 and good 2 for consumer A 
    def demand_A_x1(self, p1, p2):
        return self.alpha * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p1
    
    def demand_A_x2(self, p1, p2):
        return (1 - self.alpha) * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p2

    # 4. Define the demand function for good 1 and good 2 for consumer B
    def demand_B_x1(self, p1, p2):
        return self.beta * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p1

    def demand_B_x2(self, p1, p2):
        return (1 - self.beta) * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p2

    # d. Define the function that finds pareto improvements
    def pareto_improvements(self):
        # Create an empty list
        pareto_improvements = []
        # Using a nested for loop to find and define x_A1 and x_A2
        for i in range(self.N + 1):
            
            x_A1 = i / self.N
            for j in range(self.N + 1):
                
                x_A2 = j / self.N
                x_B1 = 1 - x_A1
                x_B2 = 1 - x_A2

                if self.u_A(x_A1, x_A2) >= self.u_A(self.endowment_A[0], self.endowment_A[1]) and \
                        self.u_B(x_B1, x_B2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]) and \
                        x_B1 == 1 - x_A1 and x_B2 == 1 - x_A2:
                    
                    # Storing combination of x_A1 and x_A2
                    pareto_improvements.append((x_A1, x_A2))

        # Return the list of pareto improvements
        return pareto_improvements
    
    # i. Define a function that plots the Edgeworth Box
    def plot_edgeworth_box(self):
        result = self.pareto_improvements()
        result = np.array(result)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        ax.set_xlabel("$x_1^A$")  # x-axis label
        ax.set_ylabel("$x_2^A$")  # y-axis label
        
        # Setting the limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.scatter(self.endowment_A[0], self.endowment_A[1], marker='s', color='black', label='Endowment')
        ax.scatter(result[:, 0], result[:, 1], color='blue', label='Pareto Improvements')

        ax.legend()
        
        plt.title('Market Equilibrium Allocations in the Edgeworth Box')
        plt.show()