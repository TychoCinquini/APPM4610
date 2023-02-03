# Import required python packages
from cmath import pi
import numpy as np
import math

# Define class
class Interp:
    def __init__(self, xnode, fnode, Neval):
        # Reassign values and store backups
        self.xnode = xnode
        self.fnode = fnode
        self.Neval = Neval
        self.Neval_backup = Neval

        # Extract number of nodes
        self.Nnodes = self.xnode.size

        # Create evaluation arrays
        self.xeval = np.linspace(self.xnode[0], self.xnode[-1], self.Neval)

    def monomial_expansion(self):
        # Reset evaluation array
        self.yeval = None

        # Create original Vandermonde matrix
        V = np.zeros((self.Nnodes, self.Nnodes))
        for i in np.arange(self.Nnodes):
            for j in np.arange(self.Nnodes):
                V[i][j] = math.pow(self.xnode[i], j)

        # Create f array
        f = np.resize(self.fnode, (self.Nnodes, 1))

        # Solve for coefficient array
        a = np.matmul(np.linalg.inv(V), f)

        # Rebuild Vandermonde matrix with evaluation rows
        V = None
        V = np.zeros((self.Neval, self.Nnodes))
        for i in np.arange(self.Neval):
            for j in np.arange(self.Nnodes):
                V[i][j] = math.pow(self.xeval[i], j)

        # Find interpolation polynomial
        self.yeval = np.matmul(V, a)

        # Return results
        return [self.xeval, self.yeval]

    def lagrange(self):
        # Create empty evaluation array
        self.yeval = np.zeros(self.Neval)

        for kk in range(self.Neval):
            self.lj = None
            self.lj = np.ones(self.Nnodes)

            for count in range(self.Nnodes):
                for jj in range(self.Nnodes):
                    if (jj != count):
                        self.lj[count] = self.lj[count] * (self.xeval[kk] - self.xnode[jj]) / (
                                    self.xnode[count] - self.xnode[jj])

            for jj in range(self.Nnodes):
                self.yeval[kk] = self.yeval[kk] + self.fnode[jj] * self.lj[jj]

        return [self.xeval, self.yeval]

    def newtondd(self):
        # Initialize and populate first column of divided difference matrix
        dd_matrix = np.zeros((self.Nnodes, self.Nnodes))
        for j in range(self.Nnodes):
            dd_matrix[j][0] = self.fnode[j]

        # Populate remainder of divided difference matrix
        for i in range(1, self.Nnodes):
            for j in range(self.Nnodes - i):
                dd_matrix[j][i] = (
                            (dd_matrix[j][i - 1] - dd_matrix[j + 1][i - 1]) / (self.xnode[j] - self.xnode[i + j]))

        # Initialize empty array for interpolated polynomial
        self.yeval = np.zeros(self.Neval)

        # Evaluate polynomial using newton divided differences
        for kk in range(self.Neval):
            # Evaluate the polynomial terms
            self.ptmp = None
            self.ptmp = np.zeros(self.Nnodes)
            self.ptmp[0] = 1.0
            for j in range(self.Nnodes - 1):
                self.ptmp[j + 1] = self.ptmp[j] * (self.xeval[kk] - self.xnode[j])

            # Evaluate the divided difference polynomial
            for j in range(self.Nnodes):
                self.yeval[kk] = self.yeval[kk] + dd_matrix[0][j] * self.ptmp[j]

        # Return results
        return [self.xeval, self.yeval]

    def barycentric(self):
        # Create empty evaluation array
        self.yeval = np.zeros(self.Neval)

        # Iterate over every point in evaluation array
        for i in np.arange(0, self.Neval):
            # Calculate phi
            phi = None
            phi = 1
            for j in np.arange(self.Nnodes):
                phi = phi * (self.xeval[i] - self.xnode[j])

            # Initialize summation to 0
            summation = 0

            # Iterate over every node
            for j in np.arange(self.Nnodes):
                # Calculate wj
                wj = 1
                for k in np.arange(self.Nnodes):
                    if j != k:
                        wj = wj / (self.xnode[j] - self.xnode[k])

                # Evaluate summation
                summation = summation + ((wj / (self.xeval[i] - self.xnode[j])) * self.fnode[j])

            # Evaluate polynomial
            self.yeval[i] = phi * summation

        # Return result
        return [self.xeval, self.yeval]

    def hermite(self, fprime):
        # Create empty evaluation array
        self.yeval = np.zeros(self.Neval)

        # Iterate over all points in evaluation array
        for kk in range(self.Neval):
            # Evaluate all lagrange polynomials
            lj = np.ones(self.Nnodes)
            for count in range(self.Nnodes):
                for jj in range(self.Nnodes):
                    if (jj != count):
                        lj[count] = lj[count] * (self.xeval[kk] - self.xnode[jj]) / (self.xnode[count] - self.xnode[jj])

            # Construct the lj'(xj)
            lpj = np.zeros(self.Nnodes)
            for count in range(self.Nnodes):
                for jj in range(self.Nnodes):
                    if (jj != count):
                        lpj[count] = lpj[count] + 1. / (self.xnode[count] - self.xnode[jj])

            yeval = 0.

            for jj in range(self.Nnodes):
                Qj = (1. - 2. * (self.xeval[kk] - self.xnode[jj]) * lpj[jj]) * lj[jj] ** 2
                Rj = (self.xeval[kk] - self.xnode[jj]) * lj[jj] ** 2
                yeval = yeval + self.fnode[jj] * Qj + fprime[jj] * Rj

            self.yeval[kk] = yeval

        # Return results
        return [self.xeval, self.yeval]
