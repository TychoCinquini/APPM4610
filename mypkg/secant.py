import numpy as np

class Secant:
    def __init__(self,f,x0,x1,tol,Nmax,save_all):
        # Re-assign inputs
        self.f = f
        self.x0 = x0
        self.x0_backup = x0
        self.x1 = x1
        self.x1_backup = x1
        self.tol = tol
        self.Nmax = Nmax
        self.save_all = save_all

    def secant(self):
        # Reset iterative variables
        self.x0 = self.x0_backup
        self.x1 = self.x1_backup
        self.x2 = None
        x_star = None

        if self.save_all:
            self.x = np.zeros((1,1))
            self.x[0] = self.x0
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