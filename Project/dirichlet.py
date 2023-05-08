# Import required python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

# Import custom packages
from FiniteDifferences2D import FiniteDifferences2D

## TEST PROBLEM #1 - HEAT
# Define meshgrid boundary
a = 0
b = 0.5
c = 0
d = 0.5

# Define f(x)
f = lambda x, y: 0.0
f = np.vectorize(f)


# Define g(x)
def g(x, y):
    if x == 0.0:
        return 0
    elif y == 0.0:
        return 0
    elif x == 0.5:
        return 200*y
    elif y == 0.5:
        return 200*x

# Define u(x,y)
u = lambda x, y: 400*x*y
u = np.vectorize(u)

# Instantiate error vector
error = []

# Iterate over multiple number of points
for num_points in np.array([5, 10, 25, 50, 100]):
    # Define number of points along each axis
    n = num_points
    m = num_points

    # Create instance of 2D Finite Differences class
    fd1 = FiniteDifferences2D(a, b, c, d, n, m, f, g, gp=None)

    # Solve problem using Poisson BC
    wl = fd1.solveLinearSystemDirichlet()

    # Create matrices for contour plot
    [X, Y, Wl] = fd1.surfacePlotElements(wl)

    # Calculate exact solution
    U = u(X, Y)

    # Calculate average absolute error
    error.append(np.mean(np.abs(U-Wl)))

# Plot error
plt.plot(np.array([25, 100, 625, 2500, 10000]), error)
plt.xlabel("Number of Mesh Points")
plt.ylabel("Average Absolute Error")
plt.yscale('log')
plt.grid(True)
plt.show()

# Plot contour
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Wl, cmap=cm.viridis)
ax.set_xlabel("Width [m]")
ax.set_ylabel("Height [m]")
ax.set_zlabel("Temperature [°C]")
plt.show()

# Save error
np.savetxt('dirichlet1.csv', error, delimiter=',')


## TEST PROBLEM #2 - NONHOMOGENOUS
# Define meshgrid boundary
a = 0
b = 2
c = 0
d = 1

# Define f(x)
f = lambda x, y: x * math.exp(y)
f = np.vectorize(f)


# Define g(x)
def g(x, y):
    if x == 0.0:
        return 0
    elif y == 0.0:
        return x
    elif x == 2.0:
        return 2 * math.exp(y)
    elif y == 1.0:
        return x * math.exp(1)

# # Define g'(x)
# def gp(x, y):
#     if x == 0.0:
#         return math.exp(y)
#     elif y == 0.0:
#         return x * math.exp(y)
#     elif x == 2.0:
#         return math.exp(y)
#     elif y == 1.0:
#         return x * math.exp(y)

# Define u(x,y)
u = lambda x, y: x * math.exp(y)
u = np.vectorize(u)

# Instantiate error vectors
error = []

# Iterate over multiple number of points
for num_points in np.array([5, 10, 25, 50, 100]):
    # Define number of points along each axis
    n = num_points
    m = num_points

    # Create instances of 2D Finite Differences class
    fd2 = FiniteDifferences2D(a, b, c, d, n, m, f, g, gp=None)

    # Solve problem using Dirichlet BC
    wl = fd2.solveLinearSystemDirichlet()

    # Create matrices for contour plot
    [X, Y, Wl] = fd2.surfacePlotElements(wl)

    # Calculate exact solution
    U = u(X, Y)

    # Calculate average absolute error
    error.append(np.mean(np.abs(U-Wl)))

# # Calculate the order of convergence
# coeffs = np.polyfit(np.log(np.array([36, 100, 625, 2500, 10000])),np.log(error),1)
# alpha = coeffs[0]
# asymptotic_error_constant = np.power(10,coeffs[1])
# print("The order of convergence is: " + str(alpha))
# print("The asymptotic error constant is: " + str(asymptotic_error_constant))
# print("\n")

# Plot error
plt.plot(np.array([25, 100, 625, 2500, 10000]), error)
plt.xlabel("Number of Mesh Points")
plt.ylabel("Average Absolute Error")
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.show()

# Plot contour
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Wl, cmap=cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# Save error
np.savetxt('dirichlet2.csv', error, delimiter=',')