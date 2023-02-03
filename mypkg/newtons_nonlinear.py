# Import required python packages
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm 

class NewtonsNonlinear:
    def __init__(self,F_func,J_func,x0,tol,Nmax):
        # Re-assign variables
        self.F_func = F_func
        self.J_func = J_func
        self.x0 = x0
        self.tol = tol
        self.Nmax = Nmax

        # Create backup x0
        self.x0_backup = self.x0

    def Newton(self):
        # Reset values
        self.xstar = None
        self.x0 = self.x0_backup
        self.x1 = None

        for its in range(self.Nmax):
            self.J = None
            self.Jinv = None
            self.J = self.J_func(self.x0)
            self.Jinv = inv(self.J)
            
            self.F = np.zeros(3)
            self.F = self.F_func(self.x0)
            
            self.x1 = self.x0 - self.Jinv.dot(self.F)
            
            if (norm(self.x1-self.x0) < self.tol):
                self.xstar = self.x1
                return[self.xstar, its]
                
            self.x0 = self.x1
        
        self.xstar = None
        print("ERROR: Exceeded max number of iterations; root not found.")
        return[self.xstar,its]

    def LazyNewton(self):
        # Reset values
        self.xstar = None
        self.x0 = self.x0_backup
        self.x1 = None
        self.J = None
        self.Jinv = None

        self.J = self.J_func(self.x0)
        self.Jinv = inv(self.J)
        for its in range(self.Nmax):
            self.F = self.F_func(self.x0)
            self.x1 = self.x0 - self.Jinv.dot(self.F)

            if (norm(self.x1-self.x0) < self.tol):
                self.xstar = self.x1
                return[self.xstar, its]
                
            self.x0 = self.x1
        
        self.xstar = None
        print("ERROR: Exceeded max number of iterations; root not found.")
        return[self.xstar,its]

    def SlackerNewton(self):
        # Reset values
        self.xstar = None
        self.x0 = self.x0_backup
        self.x1 = None
        self.x2 = None

        # Calculate iniitial Jacobian and initial inverse Jacobian
        self.J = self.J_func(self.x0)
        self.Jinv = inv(self.J)

        # Calculate first iteration
        self.F = np.zeros(3)
        self.F = self.F_func(self.x0)
        self.x1 = self.x0 - self.Jinv.dot(self.F)

        # Test if converged
        if (norm(self.x1-self.x0) < self.tol):
            self.xstar = self.x1
            return[self.xstar, 1]

        # Calculate second iteration
        self.F = None
        self.F = self.F_func(self.x1)
        self.x2 = self.x1 - self.Jinv.dot(self.F)

        # Test if converged
        if (norm(self.x2-self.x1) < self.tol):
            self.xstar = self.x2
            return[self.xstar, 2]

        # Iterate until Nmax
        for its in range(3,self.Nmax):
            # Test to see if Jacobian needs to be recomputed
            if (norm(self.x2 - self.x1) > norm(self.x1 - self.x0)):
                # Recalculate J and Jinv
                self.J = self.J_func(self.x2)
                self.Jinv = inv(self.J)

            # Calculate next iteration
            self.x0 = self.x1
            self.x1 = self.x2
            self.F = None
            self.F = self.F_func(self.x1)
            self.x2 = self.x1 - self.Jinv.dot(self.F)

            # Test if converged
            if (norm(self.x2-self.x1) < self.tol):
                self.xstar = self.x2
                return[self.xstar, its]
        
        # Max number of iterations reached
        self.xstar = None
        print("ERROR: Exceeded max number of iterations; root not found.")
        return[self.xstar,its]
    
    def Broyden(self):
        A0 = self.J_func(self.x0)
        v = self.F_func(self.x0)
        A = inv(A0)
        s = -A.dot(v)
        xk = self.x0+s
        
        for its in range(self.Nmax):
            w = v
            v = self.F_func(xk)
            y = v-w
            z = -A.dot(y)
            p = -np.dot(s,z)
            u = np.dot(s,A)
            tmp = s+z
            tmp2 = np.outer(tmp,u)
            A = A+1./p*tmp2
            s = -A.dot(v)
            xk = xk+s
            if (norm(s)<self.tol):
                alpha = xk
                return [alpha,its]
            print(its)
        
        alpha = None
        return [alpha,its]