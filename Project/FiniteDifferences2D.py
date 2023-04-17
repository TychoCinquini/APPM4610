# Import required packages
import numpy as np
import math
import scipy


# Define class
class FiniteDifferences2D:
    def __init__(self, a, b, c, d, n, m, f, g):
        # Re-assign variables
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n = n
        self.m = m
        self.f = f
        self.g = g

        # Initialize empty variables
        self.A = None
        self.Aii = None
        self.Aib = None
        self.Bii = None
        self.Bib = None
        self.bvec = None
        self.fvec = None
        self.gvec = None
        self.wl = None

        # Calculate interval sizes
        self.h = (self.b - self.a) / self.n
        self.k = (self.d - self.c) / self.m

        # Create x and y vectors
        self.x = np.linspace(self.a, self.b, num=self.n+1)
        self.y = np.linspace(self.c, self.d, num=self.m+1)

    # Function to create Aii submatrix
    def makeAii(self):
        # Calculate length of main (0th) diagonal
        diag_length = (self.m - 1) * (self.n - 1)

        # Create 0th diagonal
        diag0 = np.ones(diag_length) * (2 * (math.pow(self.h / self.k, 2) + 1))

        # Create 1th and -1th diagonals
        diag1 = -1 * np.ones(diag_length)

        # Create remaining diagonals
        diag2 = -1 * np.ones(diag_length) * math.pow(self.h / self.k, 2)

        # Create Aii matrix
        data = np.array([diag2, diag1, diag0, diag1, diag2])
        diags = np.array([-1 * (self.n - 1), -1, 0, 1, self.n - 1])
        self.Aii = scipy.sparse.spdiags(data, diags, diag_length, diag_length).toarray()

        # Remove extra entries
        for i in np.arange(1, self.m - 1):
            self.Aii[(i * (self.n - 1)) - 1, i * (self.n - 1)] = 0
            self.Aii[i * (self.n - 1), (i * (self.n - 1)) - 1] = 0

    # Create Aib submatrix
    def makeAib(self):
        # Initialize matrix
        self.Aib = np.zeros(((self.m - 1) * (self.n - 1), ((self.n + 1) * 2) + (2 * (self.m - 1))))

        # Fill in matrix entries
        temp = -1 * math.pow(self.h / self.k, 2)
        for l in np.arange(2, self.n + 1):
            self.Aib[l - 2, l - 1] = temp
            self.Aib[-1 * (l - 1), -1 * l] = temp
        self.Aib[0, self.n + 1] = -1
        self.Aib[-1, -1 * (self.n + 2)] = -1
        counterX = self.n - 2
        counterY = self.n + 2
        for i in np.arange(1, self.m - 1):
            self.Aib[counterX, counterY] = -1
            self.Aib[counterX + 1, counterY + 1] = -1
            counterX = counterX + (self.n - 1)
            counterY = counterY + 2

    # Create Bib submatrix for Poisson BC
    def makeBibPoissons(self):
        self.Bib = np.eye(((self.n + 1) * 2) + (2 * (self.m - 1)))

    # Create f subvector
    def makefvec(self):
        self.fvec = np.zeros((self.m - 1) * (self.n - 1))
        counter = 0
        for j in np.arange(1, self.m):
            for i in np.arange(1, self.n):
                self.fvec[counter] = self.f(self.x[i], self.y[j])
                counter = counter + 1

    # Create g subvector
    def makegvec(self):
        self.gvec = np.zeros(((self.n + 1) * 2) + (2 * (self.m - 1)))
        counter = 0
        for i in np.arange(self.n + 1):
            self.gvec[counter] = self.g(self.x[i], self.y[0])
            counter = counter + 1
        for j in np.arange(1, self.m):
            self.gvec[counter] = self.g(self.x[0], self.y[j])
            counter = counter + 1
            self.gvec[counter] = self.g(self.x[-1], self.y[j])
            counter = counter + 1
        for i in np.arange(self.n + 1):
            self.gvec[counter] = self.g(self.x[i], self.y[-1])
            counter = counter + 1

    # Create linear system with Poisson BC
    def createLinearSystemPoisson(self):
        # Construct submatrices
        self.makeAii()
        self.makeAib()
        self.makeBibPoissons()

        # Construct subvectors
        self.makefvec()
        self.makegvec()

        # Create linear system
        self.A = np.zeros(((self.n + 1) * (self.m + 1), (self.n + 1) * (self.m + 1)))
        self.A[0:(self.m - 1) * (self.n - 1), 0:(self.m - 1) * (self.n - 1)] = self.Aii
        self.A[0:(self.m - 1) * (self.n - 1), ((self.m - 1) * (self.n - 1)):] = self.Aib
        self.A[((self.m - 1) * (self.n - 1)):, ((self.m - 1) * (self.n - 1)):] = self.Bib
        self.bvec = np.concatenate((self.fvec, self.gvec))

    # Solve linear system
    def solveLinearSystem(self):
        # Solve linear system
        w = np.linalg.solve(self.A, self.bvec)

        # Reorder approximation vector
        self.wl = np.zeros(len(w))
        idx_gstart = (self.m - 1) * (self.n - 1)
        self.wl[0:self.n + 1] = w[idx_gstart:idx_gstart + self.n + 1]
        gcounter = idx_gstart + self.n + 1
        wcounter = 0
        counter = self.n + 1
        for i in np.arange(1, self.m):
            self.wl[counter] = w[gcounter]
            self.wl[(counter + 1):(counter + self.n)] = w[wcounter:(wcounter + self.n - 1)]
            self.wl[counter + self.n] = w[gcounter + 1]
            gcounter = gcounter + 2
            wcounter = wcounter + self.n - 1
            counter = counter + self.n + 1
        self.wl[counter:] = w[gcounter:]

        # Return approximation
        return self.wl

    # Solve linear system with Poisson BC
    def solveLinearSystemPoisson(self):
        # Create and solve linear system using Poisson BC
        self.createLinearSystemPoisson()
        self.solveLinearSystem()

    # Display solution
    def surfacePlotElements(self):
        # Create matrices for contour plot
        X, Y = np.meshgrid(self.x, self.y)
        Wl = self.wl.reshape((self.m + 1, self.n + 1))

        # Return matrices
        return [X, Y, Wl]

