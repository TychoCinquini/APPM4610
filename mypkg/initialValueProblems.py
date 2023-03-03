# Import required python packages
import numpy as np
import math

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

    def rk45(self, hmin, hmax, tol):
        # Create solution vectors
        w = [self.w0]
        t = [self.a]

        # Initialize optimal step size to hmax
        hoptimal = hmax

        # Iterate until reach end of interval
        while t[-1] != self.b:
            # Create variable to store different between 4th and 5th order approximations
            wdiff = np.Inf

            # Reset proceed to false
            proceed = False

            # Loop until 4th and 5th order approximations are similar enough
            while not proceed:
                # Calculate k values
                k1 = hoptimal*self.f(t[-1], w[-1])
                k2 = hoptimal*self.f(t[-1]+(hoptimal/4), w[-1]+(k1/4))
                k3 = hoptimal*self.f(t[-1]+(3*hoptimal/8), w[-1]+(3*k1/32)+(9*k2/32))
                k4 = hoptimal*self.f(t[-1]+(12*hoptimal/13), w[-1]+(1932*k1/2197)-(7200*k2/2197)+(7296*k3/2197))
                k5 = hoptimal*self.f(t[-1]+hoptimal, w[-1]+(439*k1/216)-(8*k2)+(3680*k3/513)-(845*k4/4104))
                k6 = hoptimal*self.f(t[-1]+(hoptimal/2), w[-1]-(8*k1/27)+(2*k2)-(3544*k3/2565)+(1859*k4/4104)-(11*k5/40))

                # Calculate 4th order approximation
                w4 = w[-1]+(25*k1/216)+(1408*k3/2565)+(2197*k4/4101)-(k5/5)

                # Calculate 5th order approximation
                w5 = w[-1]+(16*k1/135)+(6656*k3/12825)+(28561*k4/56430)-(9*k5/50)+(2*k6/55)

                # Calculate difference in approximations
                wdiff = abs(w5-w4)

                # Update optimal step size
                if wdiff > tol:
                    hoptimal = hoptimal*math.pow(tol/(2*wdiff), 0.25)
                else:
                    # If necessary, correct optimal step size based on min and max bounds
                    if hoptimal < hmin:
                        hoptimal = hmin
                        proceed = True
                    elif hoptimal > hmax:
                        hoptimal = hmax
                        proceed = True

                    if t[-1]+hoptimal > self.b:
                        hoptimal = self.b-t[-1]
                    else:
                        proceed = True

            # Add the new time step and approximation
            t.append(t[-1]+hoptimal)
            w.append(w5)

        # Return the time and approximation vectors
        return np.array(t), np.array(w)

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


class MultiIVP:
    def __init__(self, a, b, h, W0, F):
        # Re-assign values
        self.a = a
        self.b = b
        self.h = h
        self.W0 = W0
        self.F = F

        # Calculate number of points
        self.N = int((self.b-self.a)/self.h)

        # Calculate number of variables
        self.n = np.shape(W0)[0]

    def explicit_euler(self):
        # Create solution vectors
        W = np.zeros((self.n, self.N+1))
        W[:, 0] = self.W0
        t = np.zeros(self.N+1)
        t[0] = self.a

        # Iterate across all time steps
        for i in np.arange(1, self.N+1):
            # Calculate next time step
            ti = self.a+(i-1)*self.h
            t[i] = ti+self.h

            # Calculate next value of solution
            W[:, i] = W[:, i-1]+(self.h*self.F(ti, W[:, i-1]))

        # Return results
        return t, W

    def rk(self):
        # Create solution vectors
        W = np.zeros((self.n, self.N+1))
        W[:, 0] = self.W0
        t = np.zeros(self.N+1)
        t[0] = self.a

        # Iterate across all time steps
        for i in np.arange(1, self.N+1):
            # Calculate k values
            k1 = self.h*self.F(t[i-1], W[:, i-1])
            k2 = self.h*self.F(t[i-1]+(self.h/2), W[:, i-1]+(k1/2))
            k3 = self.h*self.F(t[i-1]+(self.h/2), W[:, i-1]+(k2/2))
            k4 = self.h*self.F(t[i-1]+self.h, W[:, i-1]+k3)

            # Calculate next time step
            ti = self.a + (i-1) * self.h
            t[i] = ti + self.h

            # Calculate approximation at next time step
            W[:, i] = W[:, i-1]+((k1+(2*k2)+(2*k3)+k4)/6)

        return t, W
