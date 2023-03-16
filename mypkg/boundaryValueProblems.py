# Import required python packages
import numpy as np
import math
import scipy

# Define linear BVP class
class LBVP:
    def __init__(self, a, b, h, p, q, r):
        # Re-assign values
        self.a = a
        self.b = b
        self.h = h
        self.p = p
        self.q = q
        self.r = r

        # Initialize empty values
        self.x = None

        # Calculate step size
        self.N = int((self.b-self.a)/self.h)

    def dirichlet(self, w0, wN):
        # Create x vector
        self.x = np.linspace(self.a, self.b, num=(self.N+1))

        # Store data in a, b, and c diagonals
        diags = np.zeros((3,self.N-1))
        diags[0, :] = (-1)-((self.h/2)*self.p(self.x[2:self.N+1]))
        diags[1, :] = 2+((math.pow(self.h, 2))*self.q(self.x[1:self.N]))
        diags[2, :] = (-1)+((self.h/2)*self.p(self.x[1:self.N]))

        # Create b vector
        b = -1*math.pow(self.h, 2)*self.r(self.x[1:-1])
        b[0] = b[0]+((1+((self.h/2)*self.p(self.x[1])))*w0)
        b[-1] = b[-1]+((1-((self.h/2)*self.p(self.x[-2])))*wN)

        # Solve system using Thomas Algorithm
        w = np.zeros(self.N+1)
        w[0] = w0
        w[-1] = wN
        w[1:-1] = thomas(diags, b)

        # Return solution
        return self.x, w


# Define nonlinear BVP class
class NLBVP:
    def __init__(self, a, b, h, M, TOL, f, fy, fyp):
        # Re-assign values
        self.a = a
        self.b = b
        self.h = h
        self.M = M
        self.TOL = TOL
        self.f = f
        self.fy = fy
        self.fyp = fyp

        # Initialize empty values
        self.x = None

        # Calculate step size
        self.N = int((self.b-self.a)/self.h)

    def dirichlet(self, w0, wN):
        # Create x vector
        self.x = np.linspace(self.a, self.b, num=(self.N+1))

        # Create first guess at solution
        w = np.zeros(self.N+1)
        w[0] = w0
        w[-1] = wN

        # Flag to control whether to perform another iteration
        go_on = True

        # Iteration counter
        iter = 0

        # Continue iterating
        while go_on:
            # Increment iteration counter
            iter = iter+1

            # Construct Jacobian
            J = np.zeros((self.N+1, self.N+1))
            for i in np.arange(1, self.N):
                J[i, i] = (-2/math.pow(self.h, 2))+self.fy(self.x[i], w[i], (w[i+1]-w[i-1])/(2*self.h))
                J[i, i-1] = (1/math.pow(self.h, 2))-self.fyp(self.x[i], w[i], (w[i+1]-w[i-1])/(2*self.h))
                J[i, i+1] = (1/math.pow(self.h, 2))+self.fyp(self.x[i], w[i], (w[i+1]-w[i-1])/(2*self.h))
            J[0, 0] = 1
            J[-1, -1] = 1

            # Construct F
            F = np.zeros(self.N+1)
            F[1:-1] = self.f(self.x[1:self.N], w[1:self.N], (w[2:self.N+1]-w[0:self.N-1])/(2*self.h))
            F[0] = w0
            F[-1] = wN

            # Solve linear system
            w_new = np.linalg.solve(J, F)

            # Detect if algorithm has converged
            if np.linalg.norm(w_new-w) <= self.TOL:
                go_on = False

            # Update approx
            w = w_new

            # Detect if have exceed max number of iterations
            if iter >= self.M:
                go_on = False
                w = None

        # Return results
        return self.x, w

    def neumann(self, f, alpha, beta, diff_type):
        # Create x vector
        self.x = np.linspace(self.a, self.b, num=(self.N+1))

        # Create first guess at solution
        w = np.zeros(self.N+1)
        w[0] = alpha*self.h
        w[-1] = beta*self.h

        # Flag to control whether to perform another iteration
        go_on = True

        # Iteration counter
        iter = 0

        # Continue iterating
        while go_on:
            # Increment iteration counter
            iter = iter+1

            # Construct Jacobian
            J = np.zeros((self.N+1, self.N+1))
            for i in np.arange(1, self.N):
                J[i, i] = (-2/math.pow(self.h, 2))
                J[i, i-1] = (1/math.pow(self.h, 2))
                J[i, i+1] = (1/math.pow(self.h, 2))

            # Construct F
            F = np.zeros(self.N+1)
            F[1:-1] = math.pow(self.h, 2)*self.f(self.x[1:self.N], w[1:self.N], (w[2:self.N+1]-w[0:self.N-1])/(2*self.h))

            # Apply BC
            if diff_type == "forward_backward":
                J[0, 0] = 1
                J[-1, -1] = 1
                F[0] = alpha*self.h
                F[-1] = beta*self.h
            else:
                J[0, 0] = (-2/math.pow(self.h, 2))
                J[0, 1] = (1/math.pow(self.h, 2))
                J[self.N, self.N] = (-2/math.pow(self.h, 2))
                J[self.N, self.N-1] = (1/math.pow(self.h, 2))

            # Solve linear system
            w_new = np.linalg.solve(J, F)

            # Detect if algorithm has converged
            if np.linalg.norm(w_new-w) <= self.TOL:
                go_on = False

            # Update approx
            w = w_new

            # Detect if have exceeded max number of iterations
            if iter >= self.M:
                go_on = False
                w = None

        # Return results
        return self.x, w

# Define Helper Functions
def thomas(diags, y):
    # Extract size of A
    N = diags.shape[1]

    # Initialize c', y*, and x arrays
    c = np.zeros(N-1)
    y_star = np.zeros(N)
    x = np.zeros(N)

    # Calculate c0' and y0*
    # c[0] = A[0, 1]/A[0, 0]
    # y_star[0] = y[0]/A[0, 0]
    c[0] = diags[2, 0]/diags[1, 0]
    y_star[0] = y[0]/diags[1, 0]

    # Calculate ci' and yi*
    for i in np.arange(1, N):
        # den = (A[i, i]-(A[i, i-1]*c[i-1]))
        den = (diags[1, i])-(diags[0, i]*c[i-1])
        if i != N-1:
            # c[i] = A[i, i+1]/den
            c[i] = diags[2, i]/den
        # y_star[i] = (y[i]-(A[i, i-1]*y_star[i-1]))/den
        y_star[i] = (y[i]-(diags[0, i]*y_star[i-1]))/den

    # Solve linear system
    x[-1] = y_star[-1]
    for i in np.arange(2, N+1):
        x[-1*i] = y_star[N-i]-(c[N-i] * x[N-i+1])

    # Return solution
    return x