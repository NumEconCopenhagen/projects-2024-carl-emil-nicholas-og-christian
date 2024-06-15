# Import relevant packages
import matplotlib.pyplot as plt 
import numpy as np #
from types import SimpleNamespace
from scipy import optimize 
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


# Defines the class that are used in the project
class EdgeworthBoxClass:
    # Define the parameters of the model
    def __init__(self, alpha, beta, endowment_A, num_pairs=50): #
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

    # Define the utility and demand functions for the consumers
    
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

    # Define the function that finds pareto improvements
    def pareto_improvements(self):
        # Create an empty list
        pareto_improvements = []
        # Using a nested for loop to find and define x_A1 and x_A2
        for i in range(self.N + 1): # Using the range function to iterate over the number of allocations
            
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
    
    # Question 1 - Define a function that plots the Edgeworth Box
    def plot_edgeworth_box(self): 
        result = self.pareto_improvements()
        result = np.array(result)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        # Setting the labels for the x and y axis 
        ax.set_xlabel("$x_1^A$")  # x-axis label
        ax.set_ylabel("$x_2^A$")  # y-axis label
        
        # Setting the limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plotting the endowment with a square marker
        ax.scatter(self.endowment_A[0], self.endowment_A[1], marker='s', color='black', label='Endowment')
        ax.scatter(result[:, 0], result[:, 1], color='blue', label='Pareto Improvements')
        
        
        ax.legend() # this will show the legend in the plot
        
        plt.title('Market Equilibrium Allocations in the Edgeworth Box')
        plt.show()

     # Question 2.1 - Define a function to calculate market clearing errors
    def market_clearing_errors(self, p1_values):
        errors = []
        for p1 in p1_values:
            p2 = 1  # Numeraire price
            xA1 = self.demand_A_x1(p1, p2) # The demand for good 1 for consumer A
            xB1 = self.demand_B_x1(p1, p2) # The demand for good 1 for consumer B
            xA2 = self.demand_A_x2(p1, p2) # The demand for good 2 for consumer A
            xB2 = self.demand_B_x2(p1, p2) # The demand for good 2 for consumer B
            
            e1 = xA1 + xB1 - (self.endowment_A[0] + self.endowment_B[0]) # The error in the market clearing for good 1
            e2 = xA2 + xB2 - (self.endowment_A[1] + self.endowment_B[1]) # The error in the market clearing for good 2
            
            errors.append((p1, e1, e2)) # Append the errors to the list
        return errors # Return the list of errors
    
    # Question 2.2 - Define a function to plot market clearing errors
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

    # Question 3 - Define a function to find the market clearing price
    def market_clearing_error(self, p1):
        p2 = 1  # Numeraire price
        xA1 = self.demand_A_x1(p1, p2)
        xB1 = self.demand_B_x1(p1, p2)
        return (xA1 + xB1 - (self.endowment_A[0] + self.endowment_B[0]))**2
    
    # Find the market claring price using the minimize_scalar function
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
    
    # Question 4.a - Define a function to maximize utility for A with restricted prices in P1
    def maximize_utility_A_restricted(self, p1_values):
        max_utility_A_restricted = -np.inf
        best_p1_restricted = None

        for p1 in p1_values:
            xB1, xB2 = self.demand_B_x1(p1, 1), self.demand_B_x2(p1, 1)
            xA1, xA2 = 1 - xB1, 1 - xB2
            if self.u_B(xB1, xB2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]):
                utility_A = self.u_A(xA1, xA2)
                if utility_A > max_utility_A_restricted:
                    max_utility_A_restricted = utility_A
                    best_p1_restricted = p1

        if best_p1_restricted is not None:
            optimal_xB1_4a, optimal_xB2_4a = self.demand_B_x1(best_p1_restricted, 1), self.demand_B_x2(best_p1_restricted, 1)
            optimal_xA1_4a, optimal_xA2_4a = 1 - optimal_xB1_4a, 1 - optimal_xB2_4a
            return best_p1_restricted, max_utility_A_restricted, (optimal_xA1_4a, optimal_xA2_4a), (optimal_xB1_4a, optimal_xB2_4a)
        else:
            return None, None, None, None

    # Question 4.b - Define a function to maximize utility for A with any positive price
    def maximize_utility_A_unrestricted(self):
        def objective(p1, return_neg=True):
            xB1, xB2 = self.demand_B_x1(p1, 1), self.demand_B_x2(p1, 1)
            xA1, xA2 = 1 - xB1, 1 - xB2
            if self.u_B(xB1, xB2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]):
                if return_neg:
                    return -self.u_A(xA1, xA2)  # Negative because we're minimizing
                else:
                    return self.u_A(xA1, xA2)
            else:
                return np.inf

        result = minimize_scalar(objective, bounds=(0.01, 10), method='bounded')

        if result.success:
            optimal_p1_4b = result.x
            optimal_uA_4b = -result.fun
            optimal_xB1_4b, optimal_xB2_4b = self.demand_B_x1(optimal_p1_4b, 1), self.demand_B_x2(optimal_p1_4b, 1)
            optimal_xA1_4b, optimal_xA2_4b = 1 - optimal_xB1_4b, 1 - optimal_xB2_4b
            return optimal_p1_4b, optimal_uA_4b, (optimal_xA1_4b, optimal_xA2_4b), (optimal_xB1_4b, optimal_xB2_4b)
        else:
            return None, None, None, None

    # Question 5.a -  Define a function to maximize utility for A within the restricted set C 
    def maximize_utility_A_restricted_C(self):
        max_utility = -np.inf
        best_allocation_A = None
        best_allocation_B = None
        
        for x1 in np.linspace(0, 1, self.N + 1):
            for x2 in np.linspace(0, 1, self.N + 1):
                xA1, xA2 = x1, x2
                xB1, xB2 = 1 - xA1, 1 - xA2
                if self.u_A(xA1, xA2) >= self.u_A(self.endowment_A[0], self.endowment_A[1]) and \
                   self.u_B(xB1, xB2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]):
                    utility_A = self.u_A(xA1, xA2)
                    if utility_A > max_utility:
                        max_utility = utility_A
                        best_allocation_A = (xA1, xA2)
                        best_allocation_B = (xB1, xB2)
        
        return best_allocation_A, best_allocation_B

    # Question 5.b - Define a function to maximize utility for A with no restrictions besides the endowments
    def maximize_utility_A_no_restrictions(self):
        def negative_utility_A(x):
            xA1, xA2 = x 
            return -self.u_A(xA1, xA2)

        def constraint_func(x):
            xA1, xA2 = x
            return self.u_B(1 - xA1, 1 - xA2) - self.u_B(self.endowment_B[0], self.endowment_B[1])

        bounds = [(0, 1), (0, 1)] 
        initial_guess = [0.560, 0.853]  # Example initial guess, can be adjusted

        constraints = [{'type': 'ineq', 'fun': constraint_func}]
        result = minimize(negative_utility_A, initial_guess, bounds=bounds, constraints=constraints)

        if result.success:
            optimal_xA1, optimal_xA2 = result.x
            max_uA = -result.fun  # Since we minimized negative utility, maximum utility is -result.fun
            return (optimal_xA1, optimal_xA2), (1 - optimal_xA1, 1 - optimal_xA2), max_uA
        else:
            return None, None, None

    # Question 6.a - Define a function to maximize the total utility for society
    def maximize_total_utility(self):
        def total_utility(x):
            xA1, xA2, xB1, xB2 = x
            return -(self.u_A(xA1, xA2) + self.u_B(xB1, xB2))

        def constraint_resource(x):
            xA1, xA2, xB1, xB2 = x
            return [xA1 + xB1 - 1, xA2 + xB2 - 1]

        bounds = [(0, 1), (0, 1), (0, 1), (0, 1)] 
        initial_guess = [0.5, 0.5, 0.5, 0.5]  # Example initial guess

        constraints = [{'type': 'eq', 'fun': lambda x: constraint_resource(x)}]
        result = minimize(total_utility, initial_guess, bounds=bounds, constraints=constraints)

        if result.success:
            optimal_xA1, optimal_xA2, optimal_xB1, optimal_xB2 = result.x
            max_total_utility = -result.fun  # Since we minimized negative utility, maximum utility is -result.fun
            return (optimal_xA1, optimal_xA2), (optimal_xB1, optimal_xB2), max_total_utility
        else:
            return None, None, None

     # Question 6.b - Define a function to plot the allocations in Edgeworth Box
    def plot_edgeworth_box_with_allocations(self, allocations):
        result = self.pareto_improvements()
        result = np.array(result)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        ax.set_xlabel("$x_1^A$")  # x-axis label
        ax.set_ylabel("$x_2^A$")  # y-axis label

        # Setting the limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plot endowment with transparency
        ax.scatter(self.endowment_A[0], self.endowment_A[1], marker='s', color='black', label='Endowment', alpha=0.5)

        # Plot pareto improvements with transparency
        ax.scatter(result[:, 0], result[:, 1], color='blue', label='Pareto Improvements', alpha=0.2)

        # Plot the allocations with transparency
        for alloc in allocations:
            label, allocation_A, allocation_B = alloc 
            ax.scatter(allocation_A[0], allocation_A[1], label=label, alpha=1)
        
        ax.legend()
        plt.title('Allocations in the Edgeworth Box')
        plt.show()


    # Define a function to get the allocations from previous questions
    def get_allocations(self):
        allocations = [
            ("3", (0.373, 0.704), (0.627, 0.296)),
            ("4.a", (0.622, 0.640), (0.378, 0.360)),
            ("4.b", (0.621, 0.640), (0.379, 0.360)),
            ("5.a", (0.560, 0.853), (0.440, 0.147)),
            ("5.b", (0.576, 0.844), (0.424, 0.156)),
            ("6.a", (0.333, 0.667), (0.667, 0.333))
        ]
        return allocations


# Question 7 and 8 - Define the class RandomEndowments
class RandomEndowments: # Define the class RandomEndowments
    def __init__(self, alpha=1/3, beta=2/3): 
        self.alpha = alpha
        self.beta = beta

    def generate_random_endowments(self, seed=1993, num_samples=50): # Define a function to generate random endowments
        np.random.seed(seed) # Set the seed
        omega_A = np.random.uniform(0, 1, (num_samples, 2)) # Generate random endowments for consumer A
        omega_B = 1 - omega_A # Generate random endowments for consumer B
        return omega_A, omega_B #

    def plot_random_endowments(self, omega_A): # Define a function to plot the random endowments
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.scatter(omega_A[:, 0], omega_A[:, 1], color='green', label='$\omega_A$ samples', alpha=0.5)
        ax.set_xlabel('$\omega_{1A}$')
        ax.set_ylabel('$\omega_{2A}$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Random Set W with 50 Elements')
        ax.grid(True)
        ax.legend()
        plt.show()

    def market_equilibrium(self, omega): # Define a function to find the market equilibrium
        def objective(p):
            xA1_star = self.alpha * (omega[0] + p * omega[1]) / p # Calculate the optimal allocation for consumer A
            xB1_star = self.beta * ((1 - omega[0]) + p * (1 - omega[1])) / p # Calculate the optimal allocation for consumer B
            error = np.abs(xA1_star + xB1_star - 1) # Calculate the error
            return error 
        
        res = minimize(objective, 0.5, bounds=[(0.01, 5)])
        p1_star = res.x[0]
        xA1_star = self.alpha * (omega[0] + p1_star * omega[1]) / p1_star
        xA2_star = (1 - self.alpha) * (omega[0] + p1_star * omega[1])
        xB1_star = self.beta * ((1 - omega[0]) + p1_star * (1 - omega[1])) / p1_star
        xB2_star = (1 - self.beta) * ((1 - omega[0]) + p1_star * (1 - omega[1]))
        return xA1_star, xA2_star, xB1_star, xB2_star

    def plot_market_equilibrium(self, omega_A):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        for omega in omega_A:
            xA1_star, xA2_star, xB1_star, xB2_star = self.market_equilibrium(omega)
            ax.scatter(omega[0], omega[1], color='green', alpha=0.5)
            ax.scatter(xA1_star, xA2_star, color='blue', alpha=0.5)
        ax.set_xlabel('$\omega_{1A}$')
        ax.set_ylabel('$\omega_{2A}$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Market Equilibrium Allocations in the Edgeworth Box')
        ax.grid(True)
        ax.legend(['Initial Endowments', 'Equilibrium Allocations'])
        plt.show()
