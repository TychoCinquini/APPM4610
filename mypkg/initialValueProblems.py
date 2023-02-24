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

    def rk4(self):
        # Create solution vectors
        w = np.zeros(self.N + 1)
        w[0] = self.w0
        t = np.zeros(self.N + 1)
        t[0] = self.a

        # Iterate across all time steps
        for i in np.arange(1, self.N+1):
            # Calculate k values
            k1 = self.h*self.f(t[i-1], w[i-1])
            k2 = self.h*self.f(t[i-1]+(self.h/2), w[i-1]+(k1/2))
            k3 = self.h*self.f(t[i-1]+(self.h/2), w[i-1]+(k2/2))
            k4 = self.h*self.f(t[i-1]+self.h, w[i-1]+k3)

            # Calculate next time step
            ti = self.a + (i-1) * self.h
            t[i] = ti + self.h

            # Calculate approximation at next time step
            w[i] = w[i-1]+((k1+(2*k2)+(2*k3)+k4)/6)

        return t, w

    def adams_bashforth2(self):
        # Create solution vectors
        w = np.zeros(self.N + 1)
        w[0] = self.w0
        t = np.zeros(self.N + 1)
        t[0] = self.a

        # Call RK4 to generate "initial conditions"
        ivp = IVP(self.a, self.a+self.h, self.h, self.w0, self.f)
        t[0:2], w[0:2] = ivp.rk4()

        # Iterate over all remaining time steps
        for i in np.arange(2, self.N+1):
            # Calculate next time step
            ti = self.a + (i-1) * self.h
            t[i] = ti + self.h

            # Calculate approximation at next time step
            w[i] = w[i-1] + (self.h/2)*((3*self.f(t[i-1], w[i-1]))-self.f(t[i-2], w[i-2]))

        return t, w

    def adams_bashforth4(self):
        # Create solution vectors
        w = np.zeros(self.N + 1)
        w[0] = self.w0
        t = np.zeros(self.N + 1)
        t[0] = self.a

        # Call RK4 to generate "initial conditions"
        ivp = IVP(self.a, self.a+(3*self.h), self.h, self.w0, self.f)
        t[0:4], w[0:4] = ivp.rk4()

        # Iterate over all remaining time steps
        for i in np.arange(4, self.N+1):
            # Calculate next time step
            ti = self.a + (i-1) * self.h
            t[i] = ti + self.h

            # Calculate approximation at next time step
            w[i] = w[i-1] + (self.h/24)*((55*self.f(t[i-1], w[i-1]))-(59*self.f(t[i-2], w[i-2]))+(37*self.f(t[i-3], w[i-3]))-(9*self.f(t[i-4], w[i-4])))

        return t, w