# Import required packages
import numpy as np
import matplotlib.pyplot as plt

# Load in error data
error = np.loadtxt("TestProblem1Error.csv", delimiter=',', dtype=float)

# Plot data
for i in np.arange(error.shape[1]):
    plt.plot([25, 100, 625, 2500, 10000], error[:,i])
plt.yscale('log')
plt.grid(True)
plt.xlabel("Number of Mesh Points")
plt.ylabel("Average Absolute Error")
plt.legend(("Dirichlet","Neumann","Mixed","Robin"))
plt.show()

# Load in error data
error = np.loadtxt("TestProblem2Error.csv", delimiter=',', dtype=float)

# Plot data
for i in np.arange(error.shape[1]):
    plt.plot([25, 100, 625, 2500, 10000], error[:,i])
plt.yscale('log')
plt.grid(True)
plt.xlabel("Number of Mesh Points")
plt.ylabel("Average Absolute Error")
plt.legend(("Dirichlet","Neumann","Mixed","Robin"))
plt.show()