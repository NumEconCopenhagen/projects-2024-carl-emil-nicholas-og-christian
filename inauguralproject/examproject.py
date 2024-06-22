import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

class Problem1:
    def __init__(self, A, gamma, alpha, nu, epsilon, tau, T, w):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T
        self.w = w
    
    def firm_labor_supply(self, pj):
        return (pj * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

    def firm_output(self, lj):
        return self.A * (lj ** self.gamma)

    def firm_profits(self, pj, lj):
        return pj * self.firm_output(lj) - self.w * lj

    def c1(self, wl, T, pi1, pi2, p1):
        return self.alpha * (wl + T + pi1 + pi2) / p1

    def c2(self, wl, T, pi1, pi2, p2):
        return (1 - self.alpha) * (wl + T + pi1 + pi2) / (p2 + self.tau)

    def utility_maximization(self, l, p1, p2):
        lj1 = self.firm_labor_supply(p1)
        lj2 = self.firm_labor_supply(p2)
        y1 = self.firm_output(lj1)
        y2 = self.firm_output(lj2)
        pi1 = self.firm_profits(p1, lj1)
        pi2 = self.firm_profits(p2, lj2)
        
        c1_opt = self.c1(self.w * l, self.T, pi1, pi2, p1)
        c2_opt = self.c2(self.w * l, self.T, pi1, pi2, p2)
        
        return np.log(c1_opt ** self.alpha * c2_opt ** (1 - self.alpha)) - self.nu * (l ** (1 + self.epsilon)) / (1 + self.epsilon)

    def find_optimal_l(self, p1, p2):
        res = minimize_scalar(lambda l: -self.utility_maximization(l, p1, p2), bounds=(0, 10), method='bounded')
        return res.x

    def check_market_clearing(self, p1_values, p2_values):
        excess_demand_good1 = np.zeros((len(p1_values), len(p2_values)))
        excess_demand_good2 = np.zeros((len(p1_values), len(p2_values)))
        excess_demand_labor = np.zeros((len(p1_values), len(p2_values)))

        results = []

        for i, p1 in enumerate(p1_values):
            for j, p2 in enumerate(p2_values):
                l_opt = self.find_optimal_l(p1, p2)
                lj1 = self.firm_labor_supply(p1)
                lj2 = self.firm_labor_supply(p2)
                y1 = self.firm_output(lj1)
                y2 = self.firm_output(lj2)
                pi1 = self.firm_profits(p1, lj1)
                pi2 = self.firm_profits(p2, lj2)
                
                c1_opt = self.c1(self.w * l_opt, self.T, pi1, pi2, p1)
                c2_opt = self.c2(self.w * l_opt, self.T, pi1, pi2, p2)
                
                excess_demand_good1[i, j] = c1_opt - y1
                excess_demand_good2[i, j] = c2_opt - y2
                excess_demand_labor[i, j] = (lj1 + lj2) - l_opt
                
                results.append((p1, p2, excess_demand_good1[i, j], excess_demand_good2[i, j], excess_demand_labor[i, j]))

        return results, excess_demand_good1, excess_demand_good2, excess_demand_labor


class Problem3:
    def __init__(self, seed=2024, num_points=50): 
        self.seed = seed 
        self.num_points = num_points 
        self.rng = np.random.default_rng(seed) 
        self.X = np.round(self.rng.uniform(size=(num_points, 2)), 3) # Randomly generate the points X, rounded to 3 decimals
        self.y = np.round(self.rng.uniform(size=(2,)), 3) # Randomly generate the point y, rounded to 3 decimals
        self.A = None 
        self.B = None
        self.C = None
        self.D = None

    # Question 3.1: Finding points A, B, C, and D
    def find_A(self): 
        self.A = min([x for x in self.X if x[0] > self.y[0] and x[1] > self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return np.round(self.A, 3)

    def find_B(self): 
        self.B = min([x for x in self.X if x[0] > self.y[0] and x[1] < self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return np.round(self.B, 3)

    def find_C(self): 
        self.C = min([x for x in self.X if x[0] < self.y[0] and x[1] < self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return np.round(self.C, 3)

    def find_D(self): 
        self.D = min([x for x in self.X if x[0] < self.y[0] and x[1] > self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return np.round(self.D, 3)

    def find_all_points(self): 
        self.find_A() # Find point A
        self.find_B() # Find point B
        self.find_C() # Find point C
        self.find_D() # Find point D
        return self.A, self.B, self.C, self.D
    
    # Question 3.2: Barycentric coordinates and checking if y is inside the triangle
    def barycentric_coordinates(self, y, A, B, C):
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return np.round(r1, 3), np.round(r2, 3), np.round(r3, 3)

    def is_inside_triangle(self, r):
        return all(0 <= coord <= 1 for coord in r)

    def plot(self):
        self.find_all_points()
        plt.figure(figsize=(7, 7))
        plt.scatter(self.X[:, 0], self.X[:, 1], label='X (random points)')
        plt.scatter(self.y[0], self.y[1], color='r', label='y (random point)', zorder=5)
        plt.scatter(self.A[0], self.A[1], color='g', label='A', zorder=5)
        plt.scatter(self.B[0], self.B[1], color='b', label='B', zorder=5)
        plt.scatter(self.C[0], self.C[1], color='m', label='C', zorder=5)
        plt.scatter(self.D[0], self.D[1], color='c', label='D', zorder=5)

        triangle1 = plt.Polygon([self.A, self.B, self.C], color='g', alpha=0.2, label='Triangle ABC') 
        triangle2 = plt.Polygon([self.C, self.D, self.A], color='b', alpha=0.2, label='Triangle CDA')

        plt.gca().add_patch(triangle1)
        plt.gca().add_patch(triangle2)

        plt.xlabel('$x_1^A$', fontsize=12)
        plt.ylabel('$x_2^A$', fontsize=12)
        plt.legend()
        plt.title('Allocations for $X$, $y$ and the triangles $ABC$ and $CDA$', fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

    def print_values(self):
        return self.A, self.B, self.C, self.D

    # Question 3.3: Approximate f(y) and compare with true value
    def approximate_f_y(self):
        # Step 1: Define the function to compute the value of f(x)
        f = lambda x: x[0] * x[1]

        # Step 2: Compute the values of f at points A, B, C, and D
        f_A = f(self.A)
        f_B = f(self.B)
        f_C = f(self.C)
        f_D = f(self.D)

        # Step 3: Compute the barycentric coordinates of y with respect to triangles ABC and CDA
        r_ABC = self.barycentric_coordinates(self.y, self.A, self.B, self.C)
        r_CDA = self.barycentric_coordinates(self.y, self.C, self.D, self.A)

        # Step 4: Determine which triangle y is inside and compute the approximation of f(y)
        if self.is_inside_triangle(r_ABC):
            f_y_approx = r_ABC[0] * f_A + r_ABC[1] * f_B + r_ABC[2] * f_C
            y_approx = r_ABC[0] * self.A + r_ABC[1] * self.B + r_ABC[2] * self.C
        elif self.is_inside_triangle(r_CDA):
            f_y_approx = r_CDA[0] * f_C + r_CDA[1] * f_D + r_CDA[2] * f_A
            y_approx = r_CDA[0] * self.C + r_CDA[1] * self.D + r_CDA[2] * self.A
        else:
            f_y_approx = np.nan
            y_approx = np.nan

        # Step 5: Compute the true value of f(y)
        f_y_true = f(self.y)

        # Step 6: Round results to three decimal places
        f_y_approx = np.round(f_y_approx, 3)
        f_y_true = np.round(f_y_true, 3)
        y_approx = np.round(y_approx, 3)
        true_y = np.round(self.y, 3)

        # Step 7: Print the results
        print(f"Approximation of f(y): {f_y_approx}")
        print(f"True value of f(y): {f_y_true}")