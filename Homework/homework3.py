# Import required packages
import math
import numpy as np
import matplotlib.pyplot as plt

# Import custom packages
from mypkg.initialValueProblems import IVP
from mypkg.splines import Splines

## PROBLEM 1
# Define y(t)
y = lambda t: -1/t
y = np.vectorize(y)

# Define f(t,y)
f = lambda t, y: (1/math.pow(t, 2))-(y/t)-math.pow(y,2)
f = np.vectorize(f)

# Define ft
ft = lambda t, y: (-2/math.pow(t, 3))+(y/math.pow(t,2))
ft = np.vectorize(ft)

# Define fy
fy = lambda t, y: (-1/t)-(2*y)
fy = np.vectorize(fy)

# Define initial y(t) value
y0 = -1

# Define domain
a = 1
b = 2

# Define step size
h = 0.05

# Create instance of IVP class
ivp = IVP(a, b, h, y0, f)

# Add ft and fy to IVP class
ivp.ft = ft
ivp.fy = fy

# Approximate y(t) using 2nd order Taylor
t, w = ivp.taylor2()

# Graph approximation and exact solution
plt.figure()
plt.plot(t, w)
plt.plot(t, y(t), '--')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('2nd Order Taylor Approximation Versys Exact Solution')
plt.legend(['2nd Order Taylor', 'Exact Solution'])
plt.show()

# Calculate and graph absolute error
e = np.abs(y(t) - w)
plt.figure()
plt.plot(t, e)
plt.grid(True)
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('Absolute Error |y - w|')
plt.title('Absolute Error in 2nd Order Taylor Approximation')
plt.show()

# Define desired number of evaluation points
Neval = 1001

# Create instance of interpolation class
sp = Splines(y, t, Neval)

# Perform linear spline
tspline, yspline = sp.linear()

# Define time values we want to compare at
tvals = [1.052, 1.555, 1.978]

# Iterate across all time values for comparison
for i in np.arange(len(tvals)):
    # Find index in spline arrays corresponding to current time value
    idx = np.where(np.abs(tspline - tvals[i]) < 0.0005)[0][0]

    # Compare linear spline to actual value:
    print("Time: " + str(tvals[i]))
    print(" - Linear Approximation: " + str(yspline[idx]))
    print(" - Actual Solution: " + str(y(tvals[i])))
    print(" - Absolute Error: " + str(np.abs(y(tvals[i])-yspline[idx])))

## PROBLEM 3
# Create vector of step sizes
h = 2**(-1*np.linspace(1, 16, 16))

# Create vector of δt
dt = [0.5, 1, 2]

# Create error vectors
exact_error = np.zeros((len(h), 3))
backward_error = np.zeros((len(h), 3))
centered_error = np.zeros((len(h), 3))

# Iterate over all step sizes
for i in np.arange(len(h)):
    for j in np.arange(len(dt)):
        # Create instance of IVP class
        ivp = IVP(a, b, h[i], y0, f)

        # Add ft and fy to IVP class
        ivp.ft = ft
        ivp.fy = fy

        # Approximate y(t) using 2nd order Taylor with various methods of approximating f_t
        t, w = ivp.taylor2(ft_method="exact", dt=(dt[j]*h[i]))
        exact_error[i, j] = abs(y(t[-1])-w[-1])
        t, w = ivp.taylor2(ft_method="backward_euler", dt=(dt[j]*h[i]))
        backward_error[i, j] = abs(y(t[-1])-w[-1])
        t, w = ivp.taylor2(ft_method="centered_diff", dt=(dt[j]*h[i]))
        centered_error[i, j] = abs(y(t[-1])-w[-1])

# Plot error
plt.figure()
plt.loglog(h, exact_error[:, 0], 'b-')
plt.loglog(h, backward_error[:, 0], 'r-')
plt.loglog(h, backward_error[:, 1], 'r--')
plt.loglog(h, backward_error[:, 2], 'r:')
plt.loglog(h, centered_error[:, 0], 'g-')
plt.loglog(h, centered_error[:, 1], 'g--')
plt.loglog(h, centered_error[:, 2], 'g:')
plt.grid(True)
plt.xlabel('h')
plt.ylabel('Absolute Error at t = 2')
plt.title('Absolute Error Versus Step Size for Various ∂f/∂t Approximations')
plt.legend(['Exact', 'Backward Euler (δt=0.5h)', 'Backward Euler (δt=h)', 'Backward Euler (δt=2h)',
            'Centered Difference (δt=0.5h)', 'Centered Difference (δt=h)', 'Centered Difference (δt=2h)'])
plt.show()
