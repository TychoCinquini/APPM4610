# Import required packages
import numpy as np
import math
import matplotlib.pyplot as plt

# Import custom packages
import mypkg.initialValueProblems as IVP

# Import required python packages
import numpy as np

# Define class
class IVP:
    def __init__(self, a, b, h, w0, f):
        # Re-assign values
        self.a = a
        self.b = b
        self.h = h
        self.w0 = w0
        self.f = f

        # Calculate number of points
        self.N = int((self.b-self.a)/self.h)

    def explicit_euler(self):
        # Create solution vectors
        w = np.zeros(self.N+1)
        w[0] = self.w0
        t = np.zeros(self.N+1)
        t[0] = self.a

        # Iterate across all time steps
        for i in np.arange(1, self.N+1):
            # Calculate next time step
            ti = self.a+(i-1)*self.h
            t[i] = ti+self.h

            # Calculate next value of solution
            wi = self.f(ti, w[i-1])
            w[i] = w[i-1]+self.h*wi

        # Return results
        return (t, w)

## PROBLEM 2
# Define f(t,y)
f = lambda t, y: (1+t)/(1+y)
f = np.vectorize(f)

# Define y(t)
y = lambda t: math.sqrt(math.pow(t,2)+(2*t)+6)-1
y = np.vectorize(y)

# Define initial value
y0 = 2

# Define domain
a = 1
b = 2

# Define step size
h = 0.5

# Create instance of IVP class
ivp = IVP(a, b, h, y0, f)

# Use explicit euler to solve IVP
(t, w) = ivp.explicit_euler()

# Plot results
plt.figure()
plt.plot(t,w)
plt.plot(t,y(t))
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Explicit Euler with h = 0.5')
plt.legend(['Explicit Euler','y(t)'])
plt.show()

# Calculate error
err = np.abs(y(t) - w)

# Plot error
plt.figure()
plt.plot(t,err)
plt.yscale('log')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.title('Explicit Euler with h = 0.5')
plt.show()

print("The maximum error is: " + str(np.abs(y(t[-1])-w[-1])))

# Problem 3
# Change IVP values
b = 1+(1e-5)

# Define h array
h = np.power(10,np.linspace(-5,-10,6))

# Pre-allocate error array
err = np.zeros(len(h))

# Calculate error at each h
for i in np.arange(len(h)):
    ivp = IVP(a, b, h[i], y0, f)
    _, w = ivp.explicit_euler()
    err[i] = np.abs(y(b) - w[-1])

# Plot error
plt.loglog(h,err)
plt.xlabel('h')
plt.ylabel('Absolute Error')
plt.title('Absolute Error Versus Step Size')
plt.grid(True)
plt.show()