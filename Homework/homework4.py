# Import required packages
import numpy as np
import math
import matplotlib.pyplot as plt

# Import custom packages
from mypkg.initialValueProblems import IVP

## PROBLEM 1
# Define y(t)
y = lambda t: -1/math.log(t+1)
y = np.vectorize(y)

# Define f(t,y)
f = lambda t, y: math.pow(y, 2)/(1+t)
f = np.vectorize(f)

# Define initial conditions
a = 1
b = 2
h = 0.1
y0 = -1/math.log(2)

# Create instance of IVP class
ivp = IVP(a, b, h, y0, f)

# Calculate approximation using 2 step Adams-Bashforth
t2, w2 = ivp.adams_bashforth2()

# Calculate approximation using 4 step Adams-Bashforth
t4, w4 = ivp.adams_bashforth4()

# Plot error in approximation
plt.figure()
plt.plot(t2, np.abs(y(t2)-w2))
plt.plot(t4, np.abs(y(t4)-w4))
plt.grid(True)
plt.yscale('log')
plt.legend(('2 Step', '4 Step'))
plt.xlabel('t')
plt.ylabel('Absolute Error |y - w|')
plt.title('Absolute Error in Adams-Bashforth Approximations')
plt.show()

## PROBLEM 3
# Define f(t,y)
f = lambda t, y: 1-y
f = np.vectorize(f)

# Define domain
a = 0
b = 1

# Define step size
h = 0.1

# Calculate number of points
N = int((b-a)/h)

# Create time and approximation vectors
t1 = np.linspace(a, b, N+1)
w1 = np.zeros(N+1)

# Define initial conditions
w1[0] = 0
w1[1] = 1-math.exp(-0.1)

# Iterate over remaining time steps
for i in np.arange(2, N+1):
    w1[i] = (4*w1[i-1])-(3*w1[i-2])-(2*h*f(t1[i-2], w1[i-2]))

# Define step size
h = 0.01

# Calculate number of points
N = int((b-a)/h)

# Create time and approximation vectors
t2 = np.linspace(a, b, N+1)
w2 = np.zeros(N+1)

# Define initial conditions
w2[0] = 0
w2[1] = 1-math.exp(-0.1)

# Iterate over remaining time steps
for i in np.arange(2, N+1):
    w2[i] = (4*w2[i-1])-(3*w2[i-2])-(2*h*f(t2[i-2], w2[i-2]))

# Plot approximations
plt.figure()
plt.plot(t1, w1)
plt.xlabel('t')
plt.ylabel('Approximation')
plt.title('(b)')
plt.grid(True)
plt.show()

# Plot approximations
plt.figure()
plt.plot(t2, w2)
plt.xlabel('t')
plt.ylabel('Approximation')
plt.title('(c)')
plt.grid(True)
plt.show()
