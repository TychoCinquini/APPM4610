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

    # Create Aib submatrix for Dirichlet BC
    def makeAibDirichlet(self):
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

    # Create Aib submatrix for Neumann BC
    def makeAibNeumann(self):
        # Use Dirichlet Aib matrix function to create content of matrix
        self.makeAibDirichlet()
        content = self.Aib

        # Initialize matrix
        self.Aib = np.zeros(((self.m - 1) * (self.n - 1), ((self.n + 1) * 4) + (4 * (self.m - 1)) + 4))

        # Fill in matrix using Aib matrix
        self.Aib[:, 0:((self.n+1)*2)+(2*(self.m-1))] = content

    # Create Bib submatrix for Dirichlet BC
    def makeBibDirichlet(self):
        self.Bib = np.eye(((self.n + 1) * 2) + (2 * (self.m - 1)))

    # Create Bib submatrix for Neumann BC
    def makeBibNeumann(self):
        # Create empty Bib matrix
        self.Bib = np.zeros((((self.n+1)*2)+(2*(self.m-1))+4, ((self.n+1)*4)+(4*(self.m-1))+4))

        # Set values
        self.Bib[0:self.n+1, ((self.n+1)*2)+(2*(self.m-1)):((self.n+1)*2)+(2*(self.m-1))+(self.n+1)] = -1*np.eye(self.n+1)/(2*self.k)
        for i in np.arange(2*(self.m-1)):
            self.Bib[self.n+1+i, ((self.n+1)*2)+(2*(self.m-1))+6+i] = math.pow(-1, i+1)/(2*self.h)
        self.Bib[-5-self.n:-4, -1-self.n:] = np.eye(self.n+1)/(2*self.k)
        self.Bib[0, self.n+1] = 1/(2*self.k)
        self.Bib[self.n, (2*self.n)+1] = 1/(2*self.k)
        self.Bib[self.n+1+(2*(self.m-1)), self.n+1+(2*(self.m-2))] = -1/(2*self.k)
        self.Bib[1+(2*self.n)+(2*(self.m-1)), (2*self.n+1)+2*(self.m-2)] = -1/(2*self.k)
        self.Bib[-4, 1] = 1/(2*self.h)
        self.Bib[-3, self.n-1] = -1/(2*self.h)
        self.Bib[-4, ((self.n+1)*3)+(2*(self.m-1))] = -1/(2*self.h)
        self.Bib[-3, ((self.n+1)*3)+(2*(self.m-1))+(self.n-2)] = 1/(2*self.h)
        self.Bib[-2, self.n+2+(2*(self.m-1))] = 1/(2*self.h)
        self.Bib[-1, (2*(self.n+1))+(2*(self.m-1))-2] = -1/(2*self.h)
        self.Bib[-2, -2*self.n] = -1/(2*self.h)
        self.Bib[-1, -(self.n+2)] = 1/(2*self.h)

        bottom = np.zeros(((2*(self.n+1))+(2*(self.m-1)), ((self.n+1)*4)+(4*(self.m-1))+4))
        bottom[:, -((2*(self.n+1))+(2*(self.m-1))):] = np.eye((2*(self.n+1))+(2*(self.m-1)))
        self.Bib = np.concatenate((self.Bib, bottom), axis=0)

    # Create Bii submatrix for Neumann BC
    def makeBiiNeumann(self):
        # Create empty matrix
        self.Bii = np.zeros((((self.n+1)*4)+(4*(self.m-1))+4, (self.m-1)*(self.n-1)))

        # Fill in matrix
        self.Bii[1:self.n, 0:self.n-1] = np.eye(self.n-1)/(2*self.k)
        for i in np.arange(2*(self.m-1)):
            self.Bii[self.n+1+i, i] = math.pow(-1, i)/(2*self.h)
        self.Bii[(self.n+1)+(2*(self.m-1))+1:(self.n+1)+(2*(self.m-1))+1+(self.n-1), 2*(self.m-2):] = np.eye(self.n-1)/(2*self.k)

    # Create f subvector
    def makefvec(self):
        self.fvec = np.zeros((self.m - 1) * (self.n - 1))
        counter = 0
        for j in np.arange(1, self.m):
            for i in np.arange(1, self.n):
                self.fvec[counter] = -1*math.pow(self.h, 2)*self.f(self.x[i], self.y[j])
                counter = counter + 1

    # Create g subvector for Dirichlet BC
    def makegvecDirichlet(self):
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

    # Create g subvector for Neumann BC
    def makegvecNeumann(self):
        self.makegvecDirichlet()
        self.gvec = np.concatenate((self.gvec, np.array([self.g(self.x[0], self.y[0]), self.g(self.x[self.n], self.y[0]), self.g(self.x[0], self.y[self.m]), self.g(self.x[self.n], self.y[self.m])])))

    # Create linear system with Dirichlet BC
    def createLinearSystemDirichlet(self):
        # Construct submatrices
        self.makeAii()
        self.makeAibDirichlet()
        self.makeBibDirichlet()

        # Construct subvectors
        self.makefvec()
        self.makegvecDirichlet()

        # Create linear system
        self.A = np.zeros(((self.n + 1) * (self.m + 1), (self.n + 1) * (self.m + 1)))
        self.A[0:(self.m - 1) * (self.n - 1), 0:(self.m - 1) * (self.n - 1)] = self.Aii
        self.A[0:(self.m - 1) * (self.n - 1), ((self.m - 1) * (self.n - 1)):] = self.Aib
        self.A[((self.m - 1) * (self.n - 1)):, ((self.m - 1) * (self.n - 1)):] = self.Bib
        self.bvec = np.concatenate((self.fvec, self.gvec))

    # Create linear system with Neumann BC
    def createLinearSystemNeumann(self):
        # Construct submatrices
        self.makeAii()
        self.makeAibNeumann()
        self.makeBiiNeumann()
        self.makeBibNeumann()

        # Construct subvectors
        self.makefvec()
        self.makegvecNeumann()

        print(self.fvec)
        print(self.gvec)

        import matplotlib.pyplot as plt
        # print(self.Aii)
        # plt.spy(self.Aii)
        # plt.show()
        # print(self.Aib)
        # plt.spy(self.Aib)
        # plt.show()
        # print(self.Bii)
        # plt.spy(self.Bii)
        # plt.show()
        # print(self.Bib)
        # plt.spy(self.Bib)
        # plt.show()

        # Create linear system
        top = np.concatenate((self.Aii, self.Aib), axis=1)
        bottom = np.concatenate((self.Bii, self.Bib), axis=1)
        self.A = np.concatenate((top, bottom), axis=0)

        print(self.A.shape)
        plt.spy(self.A)
        plt.show()

        self.bvec = np.concatenate((self.fvec, self.gvec))

    # Solve linear system
    def solveLinearSystem(self):
        # Solve linear system
        w = np.linalg.solve(self.A, self.bvec)
        w = w[:(self.n+1)*(self.m+1)]

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

    # Solve linear system with Dirichlet BC
    def solveLinearSystemDirichlet(self):
        # Create and solve linear system using Poisson BC
        self.createLinearSystemDirichlet()
        wl = self.solveLinearSystem()
        return wl

    # Solve linear system with Neumann BC
    def solveLinearSystemNeumann(self):
        # Create and solve linear system using Neumann BC
        self.createLinearSystemNeumann()
        wl = self.solveLinearSystem()
        return wl

    # Generate variables required for a surface plot
    def surfacePlotElements(self):
        # Create matrices for contour plot
        X, Y = np.meshgrid(self.x, self.y)
        Wl = self.wl.reshape((self.m + 1, self.n + 1))

        # Return matrices
        return [X, Y, Wl]

