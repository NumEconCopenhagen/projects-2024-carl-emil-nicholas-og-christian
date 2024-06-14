# 1. Import relevant packages
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
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

    # c. Define the utility and demand functions for the consumers
    
    # 1. Define the utility function for consumer A
    def u_A(self, x_A1, x_A2):
        if x_A1 <= 0 or x_A2 <= 0:
            return -np.inf
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    # 2. Define the utility function for consumer B
    def u_B(self, x_B1, x_B2):
        if x_B1 <= 0 or x_B2 <= 0:
            return -np.inf
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

    # g. Define a function to maximize utility for A with restricted prices in P1
    def maximize_utility_A_restricted(self, p1_values):
        best_p1 = None
        max_utility = -np.inf
        for p1 in p1_values:
            allocation_A, allocation_B = self.allocation_at_price(p1)
            utility_A = self.u_A(allocation_A[0], allocation_A[1])
            if utility_A > max_utility:
                max_utility = utility_A
                best_p1 = p1
        return best_p1, max_utility



    # Define utility functions for consumers A and B
    def u_A(self, x_A1, x_A2):
        if x_A1 <= 0 or x_A2 <= 0:
            return -np.inf
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    def u_B(self, x_B1, x_B2):
        if x_B1 <= 0 or x_B2 <= 0:
            return -np.inf
        return x_B1**self.beta * x_B2**(1 - self.beta)

    # Define the demand function for consumer B
    def demand_B(self, p1, p2=1):
        xB1_star = self.beta * ((p1 * self.endowment_B[0] + p2 * self.endowment_B[1]) / p1)
        xB2_star = (1 - self.beta) * ((p1 * self.endowment_B[0] + p2 * self.endowment_B[1]) / p2)
        return xB1_star, xB2_star

    # Function to maximize utility for A with restricted prices in P1
    def maximize_utility_A_restricted(self, p1_values):
        best_p1 = None
        max_utility = -np.inf
        for p1 in p1_values:
            xB1, xB2 = self.demand_B(p1)
            xA1, xA2 = 1 - xB1, 1 - xB2
            if self.u_B(xB1, xB2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]):
                utility_A = self.u_A(xA1, xA2)
                if utility_A > max_utility:
                    max_utility = utility_A
                    best_p1 = p1
        return best_p1, max_utility

    # Function to maximize utility for A with any positive price
    def maximize_utility_A_unrestricted(self):
        def objective(p1, return_neg=True):
            xB1, xB2 = self.demand_B(p1)
            xA1, xA2 = 1 - xB1, 1 - xB2
            if self.u_B(xB1, xB2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]):
                if return_neg:
                    return -self.u_A(xA1, xA2)
                else:
                    return self.u_A(xA1, xA2)
            else:
                return np.inf

        constraints = ({'type': 'ineq', 'fun': lambda p1: p1})
        result = minimize_scalar(objective, bounds=(0.01, 10), method='bounded', options={'disp': 3})

        if result.success:
            best_p1 = result.x
            max_utility = -objective(best_p1, return_neg=False)
            return best_p1, max_utility
        else:
            return None, None

