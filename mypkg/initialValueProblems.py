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

        # Initialize empty values
        self.fy = None
        self.ft = None

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
        return t, w

    def taylor2(self, ft_method="exact", dt=0):
        # Create solution vectors
        w = np.zeros(self.N + 1)
        w[0] = self.w0
        t = np.zeros(self.N + 1)
        t[0] = self.a

        # Iterate across all time steps
        for i in range(1, self.N+1):
            # Calculate next time step
            ti = self.a + (i-1) * self.h
            t[i] = ti + self.h

            # Evaluate function at current time step
            fi = self.f(ti, w[i-1])

            # Evaluate ft
            if ft_method == "exact":
                ft = self.ft(ti, w[i-1])
            elif ft_method == "backward_euler":
                ft = self.f(t[i], w[i-1])-self.f(t[i]-dt, w[i-1])/dt
            elif ft_method == "centered_diff":
                ft = self.f(t[i]+dt, w[i-1])-self.f(t[i]-dt, w[i-1])/(2*dt)

            # Evaluate Df at current time step
            Dfi = ft+(self.fy(ti, w[i-1])*self.f(ti, w[i-1]))
            w[i] = w[i-1]+(self.h*(fi+((self.h/2)*Dfi)))

        return t, w