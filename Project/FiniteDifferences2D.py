# Import required packages
import numpy as np
import math
import scipy

# Define class
class FiniteDifferences2D:
    def __init__(self, a, b, c, d, n, m, f, g=None, gp=None):
        # Re-assign variables
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n = n
        self.m = m
        self.f = f
        self.g = g
        self.gp = gp

        # Calculate interval sizes
        self.h = (self.b - self.a) / self.n
        self.k = (self.d - self.c) / self.m

        # Create x and y vectors
        self.x = np.linspace(self.a, self.b, num=self.n+1)
        self.y = np.linspace(self.c, self.d, num=self.m+1)

    # Function to create Aii submatrix
    def makeAii(self, n, m, h, k):
        # Calculate length of main (0th) diagonal
        diag_length = (m - 1) * (n - 1)

        # Create 0th diagonal
        diag0 = np.ones(diag_length) * (2 * (math.pow(h / k, 2) + 1))

        # Create 1th and -1th diagonals
        diag1 = -1 * np.ones(diag_length)

        # Create remaining diagonals
        diag2 = -1 * np.ones(diag_length) * math.pow(h / k, 2)

        # Create Aii matrix
        data = np.array([diag2, diag1, diag0, diag1, diag2])
        diags = np.array([-1 * (n - 1), -1, 0, 1, n - 1])
        Aii = scipy.sparse.spdiags(data, diags, diag_length, diag_length).toarray()

        # Remove extra entries
        for i in np.arange(1, m - 1):
            Aii[(i * (n - 1)) - 1, i * (n - 1)] = 0
            Aii[i * (n - 1), (i * (n - 1)) - 1] = 0

        # Return matrix
        return Aii

    # Create Aib submatrix
    def makeAib(self, n, m, h, k):
        # Initialize matrix
        Aib = np.zeros(((m - 1) * (n - 1), ((n + 1) * 2) + (2 * (m - 1))))

        # Fill in matrix entries
        temp = -1 * math.pow(h / k, 2)
        for l in np.arange(2, n + 1):
            Aib[l - 2, l - 1] = temp
            Aib[-1 * (l - 1), -1 * l] = temp
        Aib[0, n + 1] = -1
        Aib[-1, -1 * (n + 2)] = -1
        counterX = n - 2
        counterY = n + 2
        for i in np.arange(1, m - 1):
            Aib[counterX, counterY] = -1
            Aib[counterX + 1, counterY + 1] = -1
            counterX = counterX + (n - 1)
            counterY = counterY + 2

        # Return matrix
        return Aib

    # Make Att submatrix
    def makeAtt(self, n, m, h, k):
        Att = self.makeAii(n+2, m+2, h, k)
        return Att

    # Make Atg matrix
    def makeAtg(self, n, m, h, k):
        # Create over-sized Atg matrix
        Atg = self.makeAib(n+2, m+2, h, k)

        # Remove columns corresponding to corner points
        Atg = np.delete(Atg, [0, n+2, -(n+3), -1], axis=1)

        # Return matrix
        return Atg

    # Create Bib submatrix
    def makeBib(self, n, m):
        Bib = np.eye(((n + 1) * 2) + (2 * (m - 1)))
        return Bib

    # Create Btt submatrix for Neumann BC
    def makeBttNeumann(self, n, m, h, k):
        Btt = np.zeros(((2*(n+1))+(2*(m+1)), (n+1)*(m+1)))
        Btt[0:n+1, n+1:2*(n+1)] = np.eye(n+1)/(2*k)
        Btt[-(n+1):, -(2*(n+1)):-(n+1)] = -1*np.eye(n+1)/(2*k)
        counterX = n+1
        counterY = 1
        for i in np.arange(m+1):
            Btt[counterX, counterY] = 1/(2*h)
            Btt[counterX+1, counterY+n-2] = -1/(2*h)
            counterX += 2
            counterY += n+1
        return Btt

    # Create Btt submatrix for Robin BC
    def makeBttRobin(self, n, m, h, k, alpha, beta):
        # Generate entries of Btt that step from central difference
        Btt = self.makeBttNeumann(n, m, h, k)
        Btt = beta*Btt

        # Fill in remaining entries
        Btt[0:n+1, 0:n+1] = alpha*np.eye(n+1)
        Btt[-(n+1):, -(n+1):] = alpha*np.eye(n+1)
        counterX = n+1
        counterY = 0
        for i in np.arange(m+1):
            Btt[counterX, counterY] = alpha
            Btt[counterX+1, counterY+n] = alpha
            counterX += 2
            counterY += n+1

        # Return result
        return Btt

    # Create Btg submatrix for Neumann BC
    def makeBtgNeumann(self, n, m, h, k):
        Btg = np.eye((2*(n+1))+(2*(m+1)))/(2*k)
        Btg[0:n+1, 0:n+1] = -1*Btg[0:n+1, 0:n+1]
        Btg[n+1:-(n+1), n+1:-(n+1)] = np.eye(2*(m+1))/(2*h)
        for i in np.linspace(n+1, n+(2*(m+1)-1), num=m+1, endpoint=True):
            Btg[int(i), int(i)] = -1*Btg[int(i), int(i)]
        return Btg

    # Create Btg submatrix for Robin BC
    def makeBtgRobin(self, n, m, h, k, beta):
        Btg = self.makeBtgNeumann(n, m, h, k)
        Btg = beta*Btg
        return Btg

    # Create f subvector
    def makefvec(self, n, m, h, x, y, f):
        fvec = np.zeros((m - 1) * (n - 1))
        counter = 0
        for j in np.arange(1, m):
            for i in np.arange(1, n):
                fvec[counter] = -1*math.pow(h, 2)*f(x[i], y[j])
                counter = counter + 1
        return fvec

    # Create g subvector
    def makegvec(self, n, m, x, y, g):
        gvec = np.zeros(((n + 1) * 2) + (2 * (m - 1)))
        counter = 0
        for i in np.arange(n + 1):
            gvec[counter] = g(x[i], y[0])
            counter = counter + 1
        for j in np.arange(1, m):
            gvec[counter] = g(x[0], y[j])
            counter = counter + 1
            gvec[counter] = g(x[-1], y[j])
            counter = counter + 1
        for i in np.arange(n + 1):
            gvec[counter] = g(x[i], y[-1])
            counter = counter + 1
        return gvec

    # Make g' vector
    def makegpvec(self, n, m, x, y, gp):
        gvec = np.zeros(((n + 1) * 2) + (2 * (m + 1)))
        for i in np.arange(n+1):
            gvec[i] = gp(x[i], y[0], "bottom")
        idx = n+1
        for i in np.arange(m+1):
            gvec[idx] = gp(x[0], y[i], "left")
            idx += 1
            gvec[idx] = gp(x[-1], y[i], "right")
            idx += 1
        for i in np.arange(n+1):
            gvec[idx] = gp(x[i], y[-1], "top")
            idx += 1
        return gvec

    # Create linear system with Dirichlet BC
    def createLinearSystemDirichlet(self):
        # Construct submatrices
        Aii = self.makeAii(self.n, self.m, self.h, self.k)
        Aib = self.makeAib(self.n, self.m, self.h, self.k)
        Bib = self.makeBib(self.n, self.m)

        # Construct subvectors
        fvec = self.makefvec(self.n, self.m, self.h, self.x, self.y, self.f)
        gvec = self.makegvec(self.n, self.m, self.x, self.y, self.g)

        # Create linear system
        A = np.zeros(((self.n + 1) * (self.m + 1), (self.n + 1) * (self.m + 1)))
        A[0:(self.m - 1) * (self.n - 1), 0:(self.m - 1) * (self.n - 1)] = Aii
        A[0:(self.m - 1) * (self.n - 1), ((self.m - 1) * (self.n - 1)):] = Aib
        A[((self.m - 1) * (self.n - 1)):, ((self.m - 1) * (self.n - 1)):] = Bib
        bvec = np.concatenate((fvec, gvec))

        # Return A and b
        return A, bvec

    # Create linear system with Neumann BC
    def createLinearSystemNeumann(self):
        # Construct submatrices
        Att = self.makeAtt(self.n, self.m, self.h, self.k)
        Atg = self.makeAtg(self.n, self.m, self.h, self.k)
        Btt = self.makeBttNeumann(self.n, self.m, self.h, self.k)
        Btg = self.makeBtgNeumann(self.n, self.m, self.h, self.k)

        # Construct A matrix
        A = np.concatenate((Att, Atg), axis=1)
        A = np.concatenate((A, np.concatenate((Btt, Btg), axis=1)), axis=0)

        # Construct subvectors
        x = np.concatenate((np.array([self.a-self.h]), self.x, np.array([self.b+self.h])))
        y = np.concatenate((np.array([self.c-self.k]), self.y, np.array([self.c+self.k])))
        fvec = self.makefvec(self.n+2, self.m+2, self.h, x, y, self.f)
        gpvec = self.makegpvec(self.n, self.m, self.x, self.y, self.gp)

        # Create b vector
        bvec = np.concatenate((fvec, gpvec))

        # Return A and b
        return A, bvec

    # Create linear system with Robin BC
    def createLinearSystemRobin(self, alpha, beta, gRobin):
        # Construct submatrices
        Att = self.makeAtt(self.n, self.m, self.h, self.k)
        Atg = self.makeAtg(self.n, self.m, self.h, self.k)
        Btt = self.makeBttRobin(self.n, self.m, self.h, self.k, alpha, beta)
        Btg = self.makeBtgRobin(self.n, self.m, self.h, self.k, beta)

        # Construct A matrix
        A = np.concatenate((Att, Atg), axis=1)
        A = np.concatenate((A, np.concatenate((Btt, Btg), axis=1)), axis=0)

        # Construct subvectors
        x = np.concatenate((np.array([self.a-self.h]), self.x, np.array([self.b+self.h])))
        y = np.concatenate((np.array([self.c-self.k]), self.y, np.array([self.c+self.k])))
        fvec = self.makefvec(self.n+2, self.m+2, self.h, x, y, self.f)
        gvec = self.makegpvec(self.n, self.m, self.x, self.y, gRobin)

        # Create b vector
        bvec = np.concatenate((fvec, gvec))

        # Return A and b
        return A, bvec

    def createLinearSystemMixed(self, dirichlet_sides):
        # Create a system assuming all boundaries are Neumann BC
        A, bvec = self.createLinearSystemNeumann()

        # Update A and b matrices and save indices of extraneous rows and columns
        indices_to_remove = np.array([])
        for side in dirichlet_sides:
            # Bottom boundary has Dirichlet BC
            if side == "bottom":
                indices_to_remove = np.concatenate((indices_to_remove, ((self.n+1)*(self.m+1))+np.arange(self.n+1)))
                for i in np.arange(self.n+1):
                    A[i, :] = np.zeros(A.shape[1])
                    A[i, i] = 1
                    bvec[i] = self.g(self.x[i], self.y[0])

            # Left boundary has Dirichlet BC
            if side == "left":
                indices_to_remove = np.concatenate((indices_to_remove, ((self.n+1)*(self.m+1))+np.linspace(self.n+1, (self.n+1)+(2*self.m), num=self.m+1)))
                counter = 0
                for i in np.arange(self.m+1):
                    A[counter, :] = np.zeros(A.shape[1])
                    A[counter, counter] = 1
                    bvec[counter] = self.g(self.x[0], self.y[i])
                    counter += (self.n+1)

            # Right boundary has Dirichlet BC
            if side == "right":
                indices_to_remove = np.concatenate((indices_to_remove, ((self.n+1)*(self.m+1))+np.linspace(self.n+2, (self.n+2)+(2*self.m), num=self.m+1)))
                counter = self.n
                for i in np.arange(self.m+1):
                    A[counter, :] = np.zeros(A.shape[1])
                    A[counter, counter] = 1
                    bvec[counter] = self.g(self.x[-1], self.y[i])
                    counter += (self.n+1)

            # Top boundary has Dirichlet BC
            if side == "top":
                indices_to_remove = np.concatenate((indices_to_remove, ((self.n+1)*(self.m+1))+np.arange((self.n+1)+(2*(self.m+1)), (2*(self.n+1))+(2*(self.m+1)))))
                counter = (self.n+1)*self.m
                for i in np.arange(self.n+1):
                    A[counter, :] = np.zeros(A.shape[1])
                    A[counter, counter] = 1
                    bvec[counter] = self.g(self.x[i], self.y[-1])
                    counter += 1
        indices_to_remove = indices_to_remove.astype('int')
        indices_to_remove = np.sort(indices_to_remove)

        # Remove extraneous indices
        A = np.delete(A, indices_to_remove, axis=0)
        A = np.delete(A, indices_to_remove, axis=1)
        bvec = np.delete(bvec, indices_to_remove)

        # Return matrices
        return A, bvec

    # Solve linear system
    def solveLinearSystemDirichletBC(self, A, bvec, n, m):
        # Solve linear system
        w = np.linalg.solve(A, bvec)

        # Reorder approximation vector
        wl = np.zeros(len(w))
        idx_gstart = (m - 1) * (n - 1)
        wl[0:self.n + 1] = w[idx_gstart:idx_gstart + n + 1]
        gcounter = idx_gstart + n + 1
        wcounter = 0
        counter = n + 1
        for i in np.arange(1, m):
            wl[counter] = w[gcounter]
            wl[(counter + 1):(counter + n)] = w[wcounter:(wcounter + n - 1)]
            wl[counter + n] = w[gcounter + 1]
            gcounter = gcounter + 2
            wcounter = wcounter + n - 1
            counter = counter + n + 1
        wl[counter:] = w[gcounter:]

        # Return approximation
        return wl

    # Solve linear system
    def solveLinearSystemNeumannBC(self, A, bvec, n, m):
        # Solve linear system
        w = np.linalg.solve(A, bvec)

        # Remove ghost nodes
        wl = w[0:(n+1)*(m+1)]

        # Return approximation
        return wl

    # Solve linear system
    def solveLinearSystemRobinBC(self, A, bvec, n, m):
        # Solve linear system
        w = np.linalg.solve(A, bvec)

        # Remove ghost nodes
        wl = w[0:(n+1)*(m+1)]

        # Return approximation
        return wl

    # Solve linear system with mixed (Dirichlet and Neumann) BC
    def solveLinearSystemMixedBC(self, A, bvec, n, m):
        # Solve linear system
        w = np.linalg.solve(A, bvec)

        # Remove ghost nodes
        wl = w[0:(n+1)*(m+1)]

        # Return approximation
        return wl

    # Solve linear system with Dirichlet BC
    def solveLinearSystemDirichlet(self):
        # Create and solve linear system using Dirichlet BC
        A, bvec = self.createLinearSystemDirichlet()
        wl = self.solveLinearSystemDirichletBC(A, bvec, self.n, self.m)
        return wl

    # Solve linear system with Neumann BC
    def solveLinearSystemNeumann(self):
        A, bvec = self.createLinearSystemNeumann()
        wl = self.solveLinearSystemNeumannBC(A, bvec, self.n, self.m)
        return wl

    # Solve linear system with Robin BC
    def solveLinearSystemRobin(self, alpha, beta):
        # Create linear combination of g and g'
        def gRobin(x, y, side):
            return (alpha*self.g(x,y)) + (beta*self.gp(x, y, side))

        A, bvec = self.createLinearSystemRobin(alpha, beta, gRobin)

        wl = self.solveLinearSystemRobinBC(A, bvec, self.n, self.m)

        return wl

    # Solve linear system with mixed BC
    def solveLinearSystemMixed(self, dirichlet_sides):
        A, bvec = self.createLinearSystemMixed(dirichlet_sides)
        wl = self.solveLinearSystemMixedBC(A, bvec, self.n, self.m)
        return wl

    # Generate elements necessary for displaying solution
    def surfacePlotElements(self, wl):
        # Create matrices for contour plot
        X, Y = np.meshgrid(self.x, self.y)
        Wl = wl.reshape((self.m + 1, self.n + 1))

        # Return matrices
        return [X, Y, Wl]