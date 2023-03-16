# Import required python packages
import numpy as np
import math
import matplotlib.pyplot as plt

# Import custom packages
from mypkg.boundaryValueProblems import LBVP, NLBVP

## PROBLEM 1
# Define y
y = lambda x: math.log(x)
y = np.vectorize(y)

# Define f(x,y,y')
f = lambda x, y, yp: -1*math.exp(-2*y)
f = np.vectorize(f)

# Define fy
fy = lambda x, y, yp: 2*math.exp(-2*y)
fy = np.vectorize(fy)

# Define fy'
fyp = lambda x, y, yp: 0
fyp = np.vectorize(fyp)

# Define domain
a = 1
b = 2

# Define dirichlet BC
w0 = 0
wN = math.log(2)

# Calculate h
h = (b-a)/9

# Create instance of nonlinear BVP class
nlbvp = NLBVP(a, b, h, 50, math.pow(10, -4), f, fy, fyp)

# Solve system
x9, w9 = nlbvp.dirichlet(w0, wN)

# Calculate h
h = (b-a)/18

# Create instance of nonlinear BVP class
nlbvp = NLBVP(a, b, h, 50, math.pow(10, -4), f, fy, fyp)

# Solve system
x18, w18 = nlbvp.dirichlet(w0, wN)

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x9, np.abs((y(x9)-w9)/y(x9)), linewidth=2)
ax1.plot(x18, np.abs((y(x18)-w18)/y(x18)), '--')
ax1.set_ylabel("Relative Error")
ax1.legend(("N = 9", "N = 18"))
ax1.grid(True)
ax2.plot(x9, np.abs((y(x9)-w9)), linewidth=2)
ax2.plot(x18, np.abs((y(x18)-w18)), '--')
ax2.set_ylabel("Absolute Error")
ax2.set_xlabel("Time [s]")
ax2.legend(("N = 9", "N = 18"))
ax2.grid(True)
plt.suptitle("Performance of Nonlinear BVP For Differing Number of Nodes")
plt.show()

## PROBLEM 2
# Define y(x)
y = lambda x: math.exp(2*x)
y = np.vectorize(y)

# Define r(x)
r = lambda x: 4*math.exp(2*x)
r = np.vectorize(r)

# Define p(x)
p = lambda x: 0
p = np.vectorize(p)

# Define q(x)
q = lambda x: 0
q = np.vectorize(q)

# Define domain
a = 0
b = 1

# Define dirichlet BC
w0 = 1
wN = math.exp(2)

# Define step sizes
hs = np.power(10., [-2, -3, -4, -5])

# Create array to store errors
error = []

# Iterate over all step sizes
for h in hs:
    # Create instance of Linear BVP class
    lbvp = LBVP(a, b, h, p, q, r)

    # Solve system
    x, w = lbvp.dirichlet(w0, wN)

    # Calculate norm of absolute error
    error.append(np.linalg.norm(np.abs(y(x)-w)))
error = np.array(error)

# Create plot
plt.figure()
plt.plot(hs, error, linewidth=2)
plt.grid(True)
plt.yscale('log')
plt.xlim(max(hs), min(hs))
plt.xlabel('Step Size, h')
plt.ylabel('Absolute Error, |y - w|')
plt.title('Absolute Error Versus Step Size')
plt.show()


# PROBLEM 3
# Define f(x,y,y')
f = lambda x, y, yp: (2*y)-(math.pow(math.pi, 2)*math.sin(math.pi*x))-(2*math.sin(math.pi*x))
f = np.vectorize(f)

# Define fy
fy = lambda x, y, yp: 0
fy = np.vectorize(fy)

# Define fy'
fyp = lambda x, y, yp: 0
fyp = np.vectorize(fyp)

# Define domain
a = 0
b = 1

# Define neumann BC
alpha = math.pi
beta = -1*math.pi

# Define hs
hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]

# create error storage
error_fb = []
error_c = []

# Iterate over very h
for h in hs:
    # Create instance of nonlinear BVP class
    nlbvp = NLBVP(a, b, h, 50, math.pow(10, -4), f, fy, fyp)

    # Solve system
    xfb, wfb = nlbvp.neumann(f, alpha, beta, diff_type="forward_backward")
    xc, wc = nlbvp.neumann(f, alpha, beta, diff_type="centered")

    # Calculate error at x = 0.5
    idx_fb = np.where(xfb == 0.5)[0][0]
    idx_c = np.where(xc == 0.5)[0][0]
    error_fb.append(abs(math.sin(math.pi*0.5)-wfb[idx_fb]))
    error_c.append(abs(math.sin(math.pi*0.5)-wc[idx_c]))

# Create plot
plt.figure()
plt.plot(hs, error_fb, linewidth=2)
plt.plot(hs, error_c, linewidth=2)
plt.grid(True)
plt.xlim(max(hs), min(hs))
plt.xlabel('Step Size, h')
plt.ylabel('Absolute Error @ x=0.5, |y - w|')
plt.title('Absolute Error Versus Step Size')
plt.legend(("Forward/Backward","Centered"))
plt.show()