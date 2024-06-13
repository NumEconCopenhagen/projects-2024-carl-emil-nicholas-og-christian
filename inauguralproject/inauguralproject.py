# 1. Import relevant packages
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from scipy import optimize 
from scipy.optimize import minimize_scalar

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

     # e. Define a function to calculate market clearing errors
    def market_clearing_errors(self, p1_values):
        errors = []
        for p1 in p1_values:
            p2 = 1  # Numeraire price
            xA1 = self.demand_A_x1(p1, p2)
            xB1 = self.demand_B_x1(p1, p2)
            xA2 = self.demand_A_x2(p1, p2)
            xB2 = self.demand_B_x2(p1, p2)
            
            e1 = xA1 + xB1 - (self.endowment_A[0] + self.endowment_B[0])
            e2 = xA2 + xB2 - (self.endowment_A[1] + self.endowment_B[1])
            
            errors.append((p1, e1, e2))
        return errors
    
    # f. Define a function to plot market clearing errors
    def plot_market_clearing_errors(self, p1_values):
        errors = self.market_clearing_errors(p1_values)
        errors = np.array(errors)
        
        plt.figure(figsize=(10, 6))
        plt.plot(errors[:, 0], errors[:, 1], label='Error in good 1 market clearing')
        plt.plot(errors[:, 0], errors[:, 2], label='Error in good 2 market clearing', color='orange')
        plt.xlabel('$p_1$')
        plt.ylabel('Error')
        plt.title('Market Clearing Errors')
        plt.legend()
        plt.grid(True)
        plt.show()


    def market_clearing_error(self, p1):
        p2 = 1  # Numeraire price
        xA1 = self.demand_A_x1(p1, p2)
        xB1 = self.demand_B_x1(p1, p2)
        return (xA1 + xB1 - (self.endowment_A[0] + self.endowment_B[0]))**2
    
    def find_market_clearing_price(self):
        result = minimize_scalar(self.market_clearing_error, bounds=(0.5, 2.5), method='bounded')
        return result.x, result.fun
    
    def allocation_at_price(self, p1):
        p2 = 1  # Numeraire price
        xA1 = self.demand_A_x1(p1, p2)
        xA2 = self.demand_A_x2(p1, p2)
        xB1 = self.demand_B_x1(p1, p2)
        xB2 = self.demand_B_x2(p1, p2)
        return (xA1, xA2), (xB1, xB2)
    

    # g. Define a function to maximize utility for A with restricted prices in P1
    def maximize_utility_A_restricted(self, p1_values):
        best_p1 = None
        max_utility = -np.inf
        for p1 in p1_values:
            xB1 = self.demand_B_x1(p1, 1)
            xB2 = self.demand_B_x2(p1, 1)
            xA1 = 1 - xB1
            xA2 = 1 - xB2
            utility_A = self.u_A(xA1, xA2)
            if utility_A > max_utility:
                max_utility = utility_A
                best_p1 = p1
        return best_p1, max_utility

    # h. Define a function to maximize utility for A with any positive price
    def maximize_utility_A_unrestricted(self):
        def negative_utility_A(p1):
            xB1 = self.demand_B_x1(p1, 1)
            xB2 = self.demand_B_x2(p1, 1)
            xA1 = 1 - xB1
            xA2 = 1 - xB2
            return -self.u_A(xA1, xA2)
        
        result = minimize_scalar(negative_utility_A, bounds=(0.01, 10), method='bounded')
        best_p1 = result.x
        max_utility = -result.fun
        return best_p1, max_utility