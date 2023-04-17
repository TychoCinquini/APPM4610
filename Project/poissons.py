# Import required python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

# Import custom packages
from FiniteDifferences2D import FiniteDifferences2D

# Define meshgrid boundary
a = 0
b = 0.5
c = 0
d = 0.5

# Define number of points along each axis
n = 30
m = 50


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


# Create instance of 2D Finite Differences class
fd1 = FiniteDifferences2D(a, b, c, d, n, m, f, g)

# Solve problem using Poisson BC
_ = fd1.solveLinearSystemPoisson()

# Create matrices for contour plot
[X, Y, Wl] = fd1.surfacePlotElements()

# Plot contour
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Wl, cmap=cm.viridis)
ax.set_xlabel("Width [m]")
ax.set_ylabel("Height [m]")
ax.set_zlabel("Temperature [°C]")
ax.set_title("Steady-State Heat Distribution in a Thin Square Metal Plate")
plt.show()

# Define meshgrid boundary
a = 0
b = 2
c = 0
d = 1

# Define number of points along each axis
n = 50
m = 50


# Define f(x)
f = lambda x, y: x*math.exp(y)
f = np.vectorize(f)


# Define g(x)
def g(x, y):
    if x == 0.0:
        return 0
    elif y == 0.0:
        return x
    elif x == 2.0:
        return 2*math.exp(y)
    elif y == 1.0:
        return x*math.exp(1)


# Create instance of 2D Finite Differences class
fd2 = FiniteDifferences2D(a, b, c, d, n, m, f, g)

# Solve problem using Poisson BC
_ = fd2.solveLinearSystemPoisson()

# Create matrices for contour plot
[X, Y, Wl] = fd2.surfacePlotElements()

# Plot contour
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Wl, cmap=cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Approximation of f(x,y)")
plt.show()