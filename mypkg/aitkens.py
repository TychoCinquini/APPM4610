import numpy as np

class Aitkens:
    def __init__(self,p_vec,tol,Nmax):
        # Re-assign variables
        self.p_vec = p_vec
        self.tol = tol
        self.Nmax = Nmax

    # Perform Aitken's method
    def approx(self):
        # Create iterable variables
        self.pn = self.p_vec[0]
        self.pn1 = self.p_vec[1]
        self.pn2 = self.p_vec[2]

        # Create iteration vector
        self.p = np.array([self.pn, self.pn1, self.pn2])

        # Initialize counter
        self.count = 0

        # Check if third element in sequence is already suitably close to pstar
        if (np.abs(self.pn2 - self.pn1) < self.tol):
            return [self.p, self.count]

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
