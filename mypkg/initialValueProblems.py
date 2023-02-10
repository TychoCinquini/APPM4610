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