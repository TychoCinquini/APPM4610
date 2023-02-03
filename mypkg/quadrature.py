# Import required python packages
import numpy as np


# Define class
class Quad:
    def __init__(self, a, b, N, f):
        # Reassign values
        self.a = a
        self.b = b
        self.N = N
        self.f = f

        # Calculate h
        self.h = (self.b-self.a)/self.N

        # Create vector of interval endpoints
        self.x = np.linspace(self.a, self.b, num=self.N+1)

        # Initialize empty integration values
        self.int_comp_trap = None
        self.int_comp_simp = None

    def comp_trap(self):
        # Initialize output using contribution from beginning of interval
        self.int_comp_trap = self.f(self.a)

        # Initialize function eval calculator
        Neval = 1

        # Iterate over every interval
        for i in np.arange(1, self.N-1):
            self.int_comp_trap = self.int_comp_trap+(2*self.f(self.x[i]))
            Neval = Neval + 1

        # Add contribution from end of interval
        self.int_comp_trap = self.int_comp_trap+self.f(self.b)
        Neval = Neval + 1

        # Complete integral calculation
        self.int_comp_trap = self.int_comp_trap*(self.h/2)

        # Return integral
        return self.int_comp_trap, Neval

    def comp_simp(self):
        # Initialize output using contribution from beginning of interval
        self.int_comp_simp = self.f(self.a)

        # Initialize function evaluator tracker
        Neval = 1

        # Add contributions from points within interval
        for i in range(1, int((self.N/2)-1)):
            self.int_comp_simp = self.int_comp_simp+(2*self.f(self.x[2*i]))
            Neval = Neval + 1
        for i in range(1, int(self.N/2)):
            self.int_comp_simp = self.int_comp_simp+(4*self.f(self.x[(2*i)-1]))
            Neval = Neval + 1

        # Add contribution from end of interval
        self.int_comp_simp = self.int_comp_simp+self.f(self.b)
        Neval = Neval + 1

        # Complete integral calculation
        self.int_comp_simp = self.int_comp_simp*(self.h/3)

        # Return integral
        return self.int_comp_simp, Neval
