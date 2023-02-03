# Import required packages
import math
import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 2
# Define u(x,t)
u = lambda x, t, theta: math.exp((-1*K*math.pow(theta, 2)*t)/(p*c))*math.sin(theta*x)
u = np.vectorize(u)

# Declare constants
K = 2.37  # [W/(cm-K)]
p = 2.70  # [g/cm^3]
c = 0.897  # [J/(g-K)]
l = 10  # [cm]

# Define number of points in x and t vectors
N = 100

# Create x and t vectors
x = np.linspace(0, l, num=N)
t = np.linspace(0, 20, num=N)

# Create meshgrid
X, T = np.meshgrid(x, t)

# Set theta
k = 1
theta = (k*math.pi)/l

# Evaluate function
U = u(X, T, theta)
U2 = u(X, T, theta)

# Graph results
fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, T, U, 50, cmap='autumn')
ax.set_xlabel('x [cm]')
ax.set_ylabel('t [sec]')
ax.set_zlabel('u(x,t)')
ax.set_title('u(x,t) for θ=π/l')
plt.show()

# PROBLEM 4
# Define f(x) and f''(x)
f = lambda x: math.exp(x)
f2 = lambda x: math.exp(x)

# Set x0
x0 = (7*math.pi)/8

# Define centered difference approximations
C1 = lambda h: (f(x0+h)-(2*f(x0))+f(x0-h))/math.pow(h, 2)
C1 = np.vectorize(C1)
C2 = lambda h: ((-1*f(x0+(2*h)))+(16*f(x0+h))-(30*f(x0))+(16*f(x0-h))-f(x0-(2*h)))/(12*math.pow(h, 2))
C2 = np.vectorize(C2)

# Define h vector
h = 10**(-np.linspace(0, 16, 17))

# Calculate relative error for both centered difference approximations
E1 = np.abs((f2(x0)-C1(h))/f2(x0))
E2 = np.abs((f2(x0)-C2(h))/f2(x0))

# Plot results
fig2 = plt.figure()
plt.loglog(h,E1)
plt.loglog(h,E2)
plt.xlabel('h')
plt.ylabel('Relative Error')
plt.title('Relative Error Versus Step Size for C1 and C2')
plt.grid(True)
plt.legend(['C1','C2'])
plt.show()

# Define function to calculate order of convergence (slope of log-log plot)
def calcOrderOfConvergence(x,y):
    # Extract beginning and end of input arrays
    x1 = x[0]
    x2 = x[-1]
    y1 = y[0]
    y2 = y[-1]

    # Calculate and return order of convergence
    return math.log10(y2/y1)/math.log10(x2/x1)


# # Calculate order of convergence for centered different formulas
alpha1 = calcOrderOfConvergence(h[0:4], E1[0:4])
alpha2 = calcOrderOfConvergence(h[0:3], E2[0:3])
print("The order of convergence of C1 is: " + str(alpha1))
print("The order of convergence of C2 is: " + str(alpha2))
