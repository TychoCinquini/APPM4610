import numpy as np

class Iteration1D:
    def __init__(self,f,method):
        # Define self attributes
        self.f = f
        self.method = method

        # Initialize initial interval for bisection
        self.a = None
        self.b = None

        # Initialize initial guess
        self.p0 = None

        # Initialize tolerance and max iterations
        self.tol = None
        self.Nmax = None

        # Initialize info message
        self.info = None

        # Initialize root
        self.pstar = None

        # Initialize iters for Newton or fixedpt
        self.p_iters = None

        # Initialize save_all variables
        self.save_all = False

        # Initialize vector of all guesses
        self.p = None

        # Initialize error type to absolute by default
        self.error = 'absolute'

    # Define root-finding functions
    def root(self):
        # Reset self.info
        self.info = None

        if self.method == 'bisection':
            if self.a is None:
                self.info = "ERROR: Initial interval not fully defined. Please define a valid value for the beginning of the interval, a"
                self.pstar = None
            elif self.b is None:
                self.info = "ERROR: Initial interval not fully defined. Please define a valid value for the end of the interval, b"
                self.pstar = None
            elif self.tol is None:
                self.info = "ERROR: tolerance not defined. Please define a valid value for tol"
                self.pstar = None
            elif self.Nmax is None:
                self.info = "ERROR: Max iterations not specified. Please define Nmax"
                self.pstar = None
            else:
                # Run bisection method
                [self.pstar, self.ier] = bisection(self.f, self.a, self.b, self.tol, self.Nmax, self.error)

                # Classify error message
                if self.ier == 1:
                    self.info = "ERROR: No root in initial interval"
                    self.pstar = None
                elif self.ier == 2:
                    self.info = "ERROR: Exceeded max iterations. Root not found to specified tolerance"
                    self.pstar = None
                elif self.ier == 3:
                    self.info = "ERROR: Root not found"
                    self.pstar = None
                elif self.ier == 4:
                    self.info = "ERROR: Invalid error type"
                    self.pstar = None
        elif self.method == 'fixedpt':
            if self.p0 is None:
                self.info = "ERROR: No initial guess. Please define x0"
                self.pstar = None
            elif self.tol is None:
                self.info = "ERROR: tolerance not defined. Please define a valid value for tol"
                self.pstar = None
            elif self.Nmax is None:
                self.info = "ERROR: Max iterations not specified. Please define Nmax"
                self.pstar = None
            else:
                # Run fixed point method
                if self.save_all:
                    [self.pstar,self.ier,self.p] = fixedpt(self.f,self.p0,self.tol,self.Nmax,self.save_all,self.error)
                else:
                    [self.pstar,self.ier] = fixedpt(self.f,self.p0,self.tol,self.Nmax,self.save_all,self.error)

                # Classify error message
                if self.ier == 1:
                    self.info = "ERROR: Exceeded max iterations. Root not found to specified tolerance"
                    self.pstar = None
                if self.ier == 2:
                    self.info = "ERROR: Invalid error type"
                    self.pstar = None
        else:
            self.info = "ERROR: Incorrect method type. Please select either 'bisection' or 'fixedpt.'"
            self.pstar = None
        
        # Print error statement if it exists
        if self.info is not None:
            print(self.info)
        
        # Return the root
        return self.pstar

def bisection(f,a,b,tol,Nmax,error):
    '''
    Inputs:
    f,a,b       - function and endpoints of initial interval
    tol, Nmax   - bisection stops when interval length < tol
                - or if Nmax iterations have occured
    error       - error type
    Returns:
    astar - approximation of root
    ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    '''

    '''     first verify there is a root we can find in the interval '''
    ier = 0
    fa = f(a); fb = f(b)
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]

    ''' verify end point is not a root '''
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]

    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    while (count < Nmax):
        c = 0.5*(a+b)
        fc = f(c)

        if (fc == 0):
            astar = c
            ier = 0
            print("The algorithm converged in " + str(count) + " iterations!")
            return [astar, ier]

        if (fa*fc<0):
            b = c
        elif (fb*fc<0):
            a = c
            fa = fc
        else:
            astar = c
            ier = 3
            return [astar, ier]

        if error == 'absolute':
            if (abs(b-a)<tol):
                astar = a
                ier =0
                print("The algorithm converged in " + str(count) + " iterations!")
                return [astar, ier]
        elif error == 'relative':
            if ((abs(b-a)/abs(a))<tol):
                astar = a
                ier =0
                print("The algorithm converged in " + str(count) + " iterations!")
                return [astar, ier]
        else:
            astar = None
            ier = 4
            return [astar, ier]
        
        count = count +1

    astar = a
    ier = 2
    return [astar,ier]

def fixedpt(f,x0,tol,Nmax,save_all,error):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    
    if save_all:
        x = np.zeros((1,1))
        x[0] = x0
        
    while (count <Nmax):
        count = count +1
        x1 = f(x0)

        if save_all:
            x = np.append(x,x1)

        if error == 'absolute':
            if (abs(x1-x0) <tol):
                xstar = x1
                ier = 0

                if save_all:
                    return [xstar,ier,x]
                else:
                    return [xstar,ier]
        elif error == 'relative':
            if ((abs(x1-x0)/abs(x0)) <tol):
                xstar = x1
                ier = 0

                if save_all:
                    return [xstar,ier,x]
                else:
                    return [xstar,ier]
        else:
            xstar = None
            ier = 2
            if save_all:
                return [xstar,ier,x]
            else:
                return [xstar,ier]
        x0 = x1

    xstar = x1
    ier = 1

    if save_all:
        return [xstar,ier,x]
    else:
        return [xstar,ier]