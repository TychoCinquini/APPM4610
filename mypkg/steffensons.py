import numpy as np
import mypkg.aitkens as Aitkens

class Steffensons:
    def __init__(self,g,a,tol,Nmax):
        # Reassign variables
        self.g = g
        self.a = a
        self.tol = tol
        self.Nmax = Nmax

    def approx(self):
        # Calculate pn
        self.pn = self.a

        # Calculate b
        self.b = self.g(self.a)

        # Calculate c
        self.c = self.g(self.b)

        # Calculate pn1
        # self.pn1 = self.a - ((np.power((self.b - self.a),2)) / (self.c - (2 * self.b) - self.a))
        self.pn1 = self.a - ((pow((self.b-self.a),2)) / (self.c - (2*self.b) + self.a))
        print(self.pn1)

        # Initialize counter
        self.count = 1

        # Store iterations
        self.p = np.array([self.pn, self.pn1])

        print(self.pn)
        print(self.pn1)

        # Check to see if pn1 converged
        if (np.abs(self.pn1 - self.pn) < self.tol):
            return [self.p, self.count]

        # Calculate pn2
        self.a = self.pn1
        self.b = self.g(self.a)
        self.c = self.g(self.b)
        self.pn2 = self.a - ((pow((self.b-self.a),2)) / (self.c - (2*self.b) + self.a))

        # Increment counter
        self.count = self.count + 1

        # Store iteration
        self.p = np.append(self.p, self.pn2)

        # Check to see if pn2 converged
        if (np.abs(self.pn2 - self.pn1) < self.tol):
            return [self.p, self.count]

        # Perform Aitken's method
        [self.p_steffensons, self.N_steffensons] = self.approx_aitkens()
        return [self.p_steffensons, self.N_steffensons]

    # Perform Aitken's method
    def approx_aitkens(self):
        # Create iterable variables
        self.pn = self.p[0]
        self.pn1 = self.p[1]
        self.pn2 = self.p[2]

        # Perform Aitken's method
        while (self.count < self.Nmax):
            # Iterate counter
            self.count = self.count + 1

            # Calculate new iterables
            self.temp = float(self.pn2)
            self.pn2 = ((self.pn * self.pn2) - np.power(self.pn1,2)) / (self.pn2 - (2 * self.pn1) + self.pn)
            self.pn = self.pn1
            self.pn1 = self.temp
            self.temp = None

            # Store next iteration
            self.p = np.append(self.p,self.pn2)

            # Check if sequence converged to pstar
            if (np.abs(self.pn2 - self.pn1) < self.tol):
                return [self.p, self.count]
        
        # Return 0s to indicate an error
        return [0,0]