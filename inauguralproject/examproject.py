import numpy as np
import matplotlib.pyplot as plt

class Problem3:
    def __init__(self, seed=2024, num_points=50): # Set the seed and the number of points
        self.seed = seed 
        self.num_points = num_points 
        self.rng = np.random.default_rng(seed) 
        self.X = self.rng.uniform(size=(num_points, 2)) # Randomly generate the points X
        self.y = self.rng.uniform(size=(2,)) # Randomly generate the point y
        self.A = None 
        self.B = None
        self.C = None
        self.D = None

    def find_A(self): # Define the method to find the point A
        self.A = min([x for x in self.X if x[0] > self.y[0] and x[1] > self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return self.A

    def find_B(self): # Define the method to find the point B
        self.B = min([x for x in self.X if x[0] > self.y[0] and x[1] < self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return self.B

    def find_C(self): # Define the method to find the point C
        self.C = min([x for x in self.X if x[0] < self.y[0] and x[1] < self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return self.C

    def find_D(self): # Define the method to find the point D
        self.D = min([x for x in self.X if x[0] < self.y[0] and x[1] > self.y[1]], 
                     key=lambda x: np.sqrt((x[0] - self.y[0])**2 + (x[1] - self.y[1])**2))
        return self.D

    def find_all_points(self): 
        self.find_A() # Find point A
        self.find_B() # Find point B
        self.find_C() # Find point C
        self.find_D() # Find point D
        return self.A, self.B, self.C, self.D

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
