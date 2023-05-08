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

# Define g'(x)
def gp(x, y, side):
    if x == 0.0 and side == "left":
        return 400*y
    elif y == 0.0 and side == "bottom":
        return 400*x
    elif x == 0.5 and side == "right":
        return 400*y
    elif y == 0.5 and side == "top":
        return 400*x

# Define u(x,y)
u = lambda x, y: 400*x*y
u = np.vectorize(u)

# Instantiate error vector
error = []
error_no_offset = []

# Iterate over multiple number of points
for num_points in np.array([5, 10, 25, 50, 100]):
    # Define number of points along each axis
    n = num_points
    m = num_points

    # Create instance of 2D Finite Differences class
    fd1 = FiniteDifferences2D(a, b, c, d, n, m, f, g=g, gp=gp)

    # Solve problem using Poisson BC
    wl = fd1.solveLinearSystemNeumann()

    # Create matrices for contour plot
    [X, Y, Wl] = fd1.surfacePlotElements(wl)

    # Calculate exact solution
    U = u(X, Y)

    # Calculate average absolute error
    error.append(np.mean(np.abs(U-Wl)))

    # Subtract offset and calculate average absolute error
    Wl_no_offest = Wl - (Wl[0] - U[0])
    error_no_offset.append(np.mean(np.abs(U-Wl_no_offest)))

# Plot error
fig, axs = plt.subplots(2)
axs[0].plot(np.array([25, 100, 625, 2500, 10000]), error)
axs[1].plot(np.array([25, 100, 625, 2500, 10000]), error_no_offset)
plt.xlabel("Number of Mesh Points")
axs[0].set_ylabel("Average Absolute Error")
axs[1].set_ylabel("Average Absolute Error")
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].grid(True)
axs[1].grid(True)
plt.show()

# Plot contour
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Wl, cmap=cm.viridis)
ax.set_xlabel("Width [m]")
ax.set_ylabel("Height [m]")
ax.set_zlabel("Temperature [°C]")
plt.show()

# Save error
np.savetxt('neumann1.csv', error_no_offset, delimiter=',')


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

# Define g'(x)
def gp(x, y, side):
    if x == 0.0 and side == "left":
        return math.exp(y)
    elif y == 0.0 and side == "bottom":
        return x * math.exp(y)
    elif x == 2.0 and side == "right":
        return math.exp(y)
    elif y == 1.0 and side == "top":
        return x * math.exp(y)

# Define u(x,y)
u = lambda x, y: x * math.exp(y)
u = np.vectorize(u)

# Instantiate error vectors
error = []
error_no_offset = []

# Iterate over multiple number of points
for num_points in np.array([5, 10, 25, 50, 100]):
    # Define number of points along each axis
    n = num_points
    m = num_points

    # Create instances of 2D Finite Differences class
    fdn = FiniteDifferences2D(a, b, c, d, n, m, f, g=g, gp=gp)

    # Solve problem using Neumann BC
    wl = fdn.solveLinearSystemNeumann()

    # Create matrices for contour plot
    [X, Y, Wl] = fdn.surfacePlotElements(wl)

    # Calculate exact solution
    U = u(X, Y)

    # Calculate average absolute error
    error.append(np.mean(np.abs(U-Wl)))

    # Subtract offset and calculate average absolute error
    Wl_no_offest = Wl - (Wl[0] - U[0])
    error_no_offset.append(np.mean(np.abs(U-Wl_no_offest)))

# # Calculate the order of convergence
# coeffs = np.polyfit(np.log(np.array([36, 100, 625, 2500, 10000])),np.log(error),1)
# alpha = coeffs[0]
# asymptotic_error_constant = np.power(10,coeffs[1])
# print("The order of convergence is: " + str(alpha))
# print("The asymptotic error constant is: " + str(asymptotic_error_constant))
# print("\n")

# Plot error
fig, axs = plt.subplots(2)
axs[0].plot(np.array([25, 100, 625, 2500, 10000]), error)
axs[1].plot(np.array([25, 100, 625, 2500, 10000]), error_no_offset)
plt.xlabel("Number of Mesh Points")
axs[0].set_ylabel("Average Absolute Error")
axs[1].set_ylabel("Average Absolute Error")
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].grid(True)
axs[1].grid(True)
plt.show()

# Plot contour
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Wl, cmap=cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# Save error
np.savetxt('neumann2.csv', error_no_offset, delimiter=',')