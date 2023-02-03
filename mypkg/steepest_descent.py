# Import required python packages
import numpy as np
import math

# Define class
class SteepestDescent:
    def __init__(self,F_func,J_func,x0,tol,Nmax):
        # Re-assign variables
        self.F_func = F_func
        self.J_func = J_func
        self.x0 = x0
        self.tol = tol
        self.Nmax = Nmax
        
        # Create backup x0
        self.x0_backup = x0
        
        # Create g(x)
        self.g_func = lambda x : math.pow(F_func(x)[0],2) + math.pow(F_func(x)[1],2) + math.pow(F_func(x)[2],2)
        
    def find_root(self):
        k = 1
        while k <= self.Nmax:
            # Calculate F
            F = self.F_func(self.x0)
            
            # Calculate J
            J = self.J_func(self.x0)
            
            # Calculate g1
            g1 = self.g_func(self.x0)
                
            # Calculate gradient
            z = 2*np.matmul((J.transpose()),F)
            z0 = np.linalg.norm(z)
            
            # Test to see if zero gradient
            if z0 == 0:
                print("ERROR: zero gradient; may have a local minima")
                return [None,None]
                
            # Make z a unit vector
            z = z/z0
            alpha1 = 0
            alpha3 = 1
            g3 = self.g_func(self.x0-(alpha3*z))
            
            while g3 >= g1:
                alpha3 = alpha3/2
                g3 = self.g_func(self.x0-(alpha3*z))
                
                if alpha3 < (self.tol/2):
                    print("ERROR: no likely improvement; may have a local minima")
                    return [None,None]
            
            alpha2 = alpha3/2
            g2 = self.g_func(self.x0-(alpha2*z))
            
            h1 = (g2-g1)/alpha2
            h2 = (g3-g2)/(alpha3-alpha2)
            h3 = (h2-h1)/alpha3
            
            alpha0 = 0.5*(alpha2-(h1/h3))
            
            g0 = self.g_func(self.x0-(alpha0*z))
            g1 = self.g_func(self.x0-(alpha1*z))
            g2 = self.g_func(self.x0-(alpha2*z))
            g3 = self.g_func(self.x0-(alpha3*z))
            g_array = np.array([g0,g1,g2,g3])
            
            idx = np.where(g_array == g_array.min())
            if idx == 0:
                alpha = alpha0
                g = g0
            elif idx == 1:
                alpha = alpha1
                g = g1
            elif idx == 2:
                alpha = alpha2
                g = g2
            else:
                alpha = alpha3
                g = g3
                
            self.x0 = self.x0-(alpha*z)
            
            if np.abs(g-g1) < self.tol:
                return [self.x0,k]
            
            k = k + 1
            
        print("ERROR: exceeded max number of iterations")
        return None