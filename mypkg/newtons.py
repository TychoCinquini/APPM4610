import numpy as np
import math

class Newtons:
    def __init__(self,f,f_prime,x0,tol,Nmax,save_all):
        # Re-assign inputs
        self.f = f
        self.f_prime = f_prime
        self.x0 = x0
        self.x0_backup = x0
        self.tol = tol
        self.Nmax = Nmax
        self.save_all = save_all

        # Initialize interval for Newton's method + bisection
        self.a = None
        self.b = None

        # Initialize second derivative for Newton's method + bisection
        self.f_double_prime = None

    def newtons(self):
        # Reset iterative variables
        self.x0 = self.x0_backup
        self.x1 = None
        x_star = None

        if self.save_all:
            self.x = np.zeros((1,1))
            self.x[0] = self.x0

        # Iterate until root is found or Nmax is reached
        for i in np.linspace(1,self.Nmax,self.Nmax):
            # Calculate next iteration
            self.x1 = self.x0-(self.f(self.x0)/self.f_prime(self.x0))

            # Store current iteration
            if self.save_all:
                self.x = np.append(self.x,self.x1)

            # Check to see if converged to root
            if np.abs(self.x1-self.x0) < self.tol:
                x_star = self.x1
                print("The algorithm converged in " + str(i) + " iterations!")
                if self.save_all:
                    return [x_star,self.x]
                else:
                    return x_star
            
            # Proceed to next time step
            self.x0 = self.x1

        # Return an error if Nmax reached
        print("ERROR: Algorithm didn't converge before max number of iterations")
        if self.save_all:
            return [None,None]
        else:
            return None

    def secant(self):
        # Reset iterative variables
        self.x0 = self.x0_backup
        self.x1 = None
        self.x2 = None
        x_star = None

        if self.save_all:
            self.x = np.zeros((1,1))
            self.x[0] = self.x0

        # Perform one step of Newton's method
        self.x1 = self.x0-(self.f(self.x0)/self.f_prime(self.x0))

        # Record iteration
        if self.save_all:
            self.x = np.append(self.x,self.x1)

        # Chck to see if converged to root
        if np.abs(self.x1-self.x0) < self.tol:
            x_star = self.x1
            print("The algorithm converged in 1 iteration!")
            if self.save_all:
                return [x_star,self.x]
            else:
                return x_star

        # Perform secant method
        for i in np.linspace(2,self.Nmax,self.Nmax-1):
            # Calculate next iteration
            self.x2 = self.x1-((self.f(self.x1)*(self.x0-self.x1))/(self.f(self.x0)-self.f(self.x1)))

            # Record iteration
            if self.save_all:
                self.x = np.append(self.x,self.x2)

            # Check to see if converged to root
            if np.abs(self.x2-self.x1) < self.tol:
                x_star = self.x2
                print("The algorithm converged in " + str(i) + " iterations!")
                if self.save_all:
                    return [x_star,self.x]
                else:
                    return x_star

            # Proceed to next time step
            self.x0 = float(self.x1)
            self.x1 = float(self.x2)

        # Return an error if Nmax reached
        print("ERROR: Algorithm didn't converge before max number of iterations")
        if self.save_all:
            return [None,None]
        else:
            return None

    def biNewtons(self):
        # Test if root in initial interval
        if self.f(self.a)*self.f(self.b) > 0:
            print("ERROR: Unable to tell if root in interval; select different interval")
            if self.save_all:
                return [None,None]
            else:
                return None

        if self.save_all:
            self.x = np.zeros((1,1))
        
        # Perform BiNewton's method
        count = 0
        while count < self.Nmax:
            # Find midpoint of interval
            self.c = (self.a+self.b)/2

            # Test to see if interval already prepared for Newton's method
            self.g = np.abs(1-(((math.pow(self.f_prime(self.c),2))-(self.f(self.c)*self.f_double_prime(self.c)))/(math.pow(self.f_prime(self.c),2))))
            if self.g < 1: # Perform Newton's method
                print("Performing biNewton's method with initial guess x0 = " + str(self.c))

                # Reset iterative variables
                self.x0 = self.c
                self.x1 = None
                x_star = None

                if self.save_all:
                    if count == 0:
                        self.x[0] = self.x0
                    else:
                        self.x = np.append(self.x,self.x0)

                # Iterate until root is found or Nmax is reached
                for i in np.linspace(count+1,self.Nmax,self.Nmax-count):
                    # Calculate next iteration
                    self.x1 = self.x0-(self.f(self.x0)/self.f_prime(self.x0))

                    # Store current iteration
                    if self.save_all:
                        self.x = np.append(self.x,self.x1)

                    # Check to see if converged to root
                    if np.abs(self.x1-self.x0) < self.tol:
                        x_star = self.x1
                        print("The algorithm converged in " + str(i) + " iterations!")
                        if self.save_all:
                            return [x_star,self.x]
                        else:
                            return x_star
                    
                    # Proceed to next time step
                    self.x0 = self.x1

                # Return an error if Nmax reached
                print("ERROR: Algorithm didn't converge before max number of iterations")
                if self.save_all:
                    return [None,None]
                else:
                    return None

            else: # Increment interval using bisection method
                if (self.f(self.a)*self.f(self.c)<0):
                    self.b = self.c
                elif (self.f(self.b)*self.f(self.c)<0):
                    self.a = self.c

            # Save current iteration
            if self.save_all:
                if count == 0:
                    self.x[0] = self.c
                else:
                    self.x = np.append(self.x,self.c)
            
            # Increment counter
            count = count + 1
        
        print("ERROR: Couldn't find a suitable interval before max number of iterations")
        if self.save_all:
            return [None,None]
        else:
            return None