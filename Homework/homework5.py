# Import required packages
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import custom packages
from mypkg.initialValueProblems import IVP, MultiIVP

## PROBLEM 4
# Define y(t)
y = lambda t: -3+(2/(1+math.exp(-2*t)))
y = np.vectorize(y)

# Define f(t,y)
f = lambda t, y: -1*(y+1)*(y+3)
f = np.vectorize(f)

# Define interval
a = 0
b = 3
h = 0.5

# Define initial condition
y0 = -2

# Define step size constraints and tolerance
hmin = 0.05
hmax = 0.5
tol = math.pow(10,-6)

# Create instance of IVP class
ivp = IVP(a, b, h, y0, f)

# Calculate approximation using Runge-Kutta-Fehlberg method
t45, w45 = ivp.rk45(hmin, hmax, tol)

# Calculate approximation using rk4 and different step sizes
t4_1, w4_1 = ivp.rk4()
h = 0.05
ivp = IVP(a, b, h, y0, f)
t4_2, w4_2 = ivp.rk4()

# Plot errors in approximations
plt.figure()
plt.plot(t45, np.abs(y(t45)-w45))
plt.plot(t4_1, np.abs(y(t4_1)-w4_1))
plt.plot(t4_2, np.abs(y(t4_2)-w4_2))
plt.grid(True)
plt.yscale('log')
plt.legend(('Runge-Kutta-Fehlberg', 'RK4 w/ h=0.5', 'RK4 w/ h=0.05'))
plt.xlabel('t')
plt.ylabel('Absolute Error |y - w|')
plt.title('Absolute Error in Approximations')
plt.show()

## PROBLEM 5
# Define F(t,X)
F = lambda t, Y: np.array([Y[1], (-2*Y[0])+(3*Y[1])+(6*np.exp(-t))])

# Define y(t)
y = lambda t: (2*math.exp(2*t))-math.exp(t)+math.exp(-t)
y = np.vectorize(y)

# Define domain
a = 0
b = 1

# Define IC
Y0 = np.array([2, 2])

# Define step size
h = 0.1

# Create instance of MultiIVP class
mivp = MultiIVP(a, b, h, Y0, F)

# Solve using euler's method
te, We = mivp.explicit_euler()
we = We[0, :]

# Solve using Runge-Kutta
trk, Wrk = mivp.rk()
wrk = Wrk[0, :]

# Solve using scipy
sol = solve_ivp(F, [a, b], Y0)
tsp = sol.t
wsp = (sol.y)[0, :]

# Plot errors in approximations
plt.figure()
plt.plot(te, np.abs(y(te)-we))
plt.plot(trk, np.abs(y(trk)-wrk))
plt.plot(tsp, np.abs(y(tsp)-wsp))
plt.grid(True)
plt.yscale('log')
plt.legend(('Explicit Euler', 'RK4', 'SciPy'))
plt.xlabel('t')
plt.ylabel('Absolute Error |y - w|')
plt.title('Absolute Error in Approximations')
plt.show()
