# Import required python packages
import numpy as np
import math

# Define class
class Splines:
    def __init__(self, f, xnode, Neval):
        # Re-define inputs as class variables
        self.f = f
        self.xnode = xnode
        self.Neval = Neval

        # Extract number of nodes
        self.Nnodes = self.xnode.size

        # Calculate number of intervals
        self.Nintervals = self.Nnodes-1

        # Create h vector
        self.h = np.zeros(self.Nintervals)
        for i in np.arange(self.Nintervals):
            self.h[i] = self.xnode[i+1] - self.xnode[i]

        # Create evaluation array
        self.xeval = np.linspace(self.xnode[0], self.xnode[-1], self.Neval)

    def natural(self):
        # Construct linear system to solve for M values
        A = np.zeros((self.Nnodes, self.Nnodes))
        A[0, 0] = 1
        for i in np.arange(1, self.Nnodes-1):
            A[i, i-1] = self.h[i-1] / 6
            A[i, i] = (self.h[i-1] + self.h[i]) / 3
            A[i, i+1] = self.h[i] / 6
        A[-1, -1] = 1

        b = np.zeros((self.Nnodes, 1))
        for i in np.arange(1, self.Nnodes-1):
            b[i, 0] = ((self.f(self.xnode[i+1]) - self.f(self.xnode[i])) / self.h[i]) - ((self.f(self.xnode[i]) - self.f(self.xnode[i-1])) / self.h[i-1])

        # Solve linear system for M values
        M = np.linalg.solve(A, b)
        M = M.reshape((1, self.Nnodes))[0]

        # Create empty yeval array
        self.yeval = np.zeros(self.Neval)

        # Iterate over every interval
        for i in np.arange(self.Nintervals):
            # Find indices of xeval in the current interval
            idx = np.where((self.xeval >= self.xnode[i]) & (self.xeval <= self.xnode[i+1]))
            idx = idx[0]

            # Iterate over every point in interval
            for j in idx:
                self.yeval[j] = eval_cubic(self.xeval[j], self.xnode[i], self.xnode[i+1], M[i], M[i+1],
                                           self.f(self.xnode[i]), self.f(self.xnode[i+1]), self.h[i])

        # Return results
        return [self.xeval, self.yeval]

    def clamped(self, fprime):
        # Construct linear system to solve for M values
        A = np.zeros((self.Nnodes, self.Nnodes))
        A[0, 0] = self.h[0] / 3
        A[0, 1] = self.h[0] / 6
        for i in np.arange(1, self.Nnodes-1):
            A[i, i-1] = self.h[i-1] / 6
            A[i, i] = (self.h[i-1] + self.h[i]) / 3
            A[i, i+1] = self.h[i] / 6
        A[-1, -1] = self.h[-1] / 3
        A[-1, -2] = self.h[-1] / 6

        b = np.zeros((self.Nnodes, 1))
        b[0, 0] = (-1 * fprime(self.xnode[0])) + ((self.f(self.xnode[1]) - self.f(self.xnode[0])) / self.h[0])
        for i in np.arange(1, self.Nnodes-1):
            b[i, 0] = ((self.f(self.xnode[i+1]) - self.f(self.xnode[i])) / self.h[i]) - ((self.f(self.xnode[i]) - self.f(self.xnode[i-1])) / self.h[i-1])
        b[-1, 0] = (-1 * fprime(self.xnode[-1])) + ((self.f(self.xnode[-1]) - self.f(self.xnode[-2])) / self.h[-1])

        # Solve linear system for M values
        M = np.linalg.solve(A, b)
        M = M.reshape((1, self.Nnodes))[0]

        # Create empty yeval array
        self.yeval = np.zeros(self.Neval)

        # Iterate over every interval
        for i in np.arange(self.Nintervals):
            # Find indices of xeval in the current interval
            idx = np.where((self.xeval >= self.xnode[i]) & (self.xeval <= self.xnode[i+1]))
            idx = idx[0]

            # Iterate over every point in interval
            for j in idx:
                self.yeval[j] = eval_cubic(self.xeval[j], self.xnode[i], self.xnode[i+1], M[i], M[i+1],
                                           self.f(self.xnode[i]), self.f(self.xnode[i+1]), self.h[i])

        # Return results
        return [self.xeval, self.yeval]

    def linear(self):
        # Create vector to store evaluation of linear splines
        self.yeval = np.zeros(self.Neval)

        for i in range(self.Nnodes-1):
            # Find indices of xeval in the current interval
            ind = np.where((self.xeval >= self.xnode[i]) & (self.xeval <= self.xnode[i+1]))
            ind = ind[0]

            # Calculate number of indices
            n = ind.size

            # Temporarily store info needed to create a line in the current interval
            a = self.xnode[i]
            b = self.xnode[i+1]

            # Evaluate line at all the points in the interval
            y = eval_line(a, b, self.f, self.xeval[ind])

            # Add points to yeval
            self.yeval[ind] = y

        # Return yeval
        return self.xeval, self.yeval



# Subroutines
def eval_cubic(x, xi, xi1, Mi, Mi1, fxi, fxi1, hi):
    # Calculate C and D coefficients
    C = (fxi / hi) - ((hi * Mi) / 6)
    D = (fxi1 / hi) - ((hi * Mi1) / 6)

    # Calculate value of spline
    Si = ((math.pow((xi1 - x), 3) * Mi) / (6 * hi)) + ((math.pow((x - xi), 3) * Mi1) / (6 * hi)) + (C * (xi1 - x)) +\
         (D * (x - xi))

    # Return result
    return Si

def eval_line(x0, x1, f, x):
    # Evaluate function at endpoints
    f0 = f(x0)
    f1 = f(x1)

    # Calculate slope of line
    m = (f1 - f0) / (x1 - x0)

    # Evaluate line
    y = (m * (x - x1)) + f1

    # Return results
    return y
