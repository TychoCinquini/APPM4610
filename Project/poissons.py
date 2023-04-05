# Import required python packages
import numpy as np
import math
import scipy

# Helper functions
def makeA(h, k, n, m):
    lmax = (m-1)*(n-1)
    diag0 = np.ones(lmax)*(2*(math.pow(h/k, 2)+1))
    diag1 = -1*np.ones(lmax)
    diag2 = -1*np.ones(lmax)*math.pow(h/k, 2)
    data = np.array([diag2, diag1, diag0, diag1, diag2])
    diags = np.array([-1*(n-1), -1, 0, 1, n-1])
    A = scipy.sparse.spdiags(data, diags, lmax, lmax).toarray()
    for i in np.arange(1, m-1):
        A[(i*(n-1))-1, i*(n-1)] = 0
        A[i*(n-1), (i*(n-1))-1] = 0
    return A

def makeb(f, g, h, k, n, m, x, y):
    lmax = (m - 1) * (n - 1)
    b = np.zeros(lmax)
    for i in np.arange(1,n):
        for j in np.arange(1,m):
            l = i+((m-1-j)*(n-1))
            b[l-1] = -1*math.pow(h, 2)*f(x[i], y[j])
    for i in np.arange(m-1):
        b[i*n] = b[i*n] + g(x[0], y[m-i-1])
        b[(n-1)*(i+1)-1] = b[(n-1)*(i+1)-1] + g(x[n], y[m-i-1])
    for i in np.arange(n-1):
        b[i] = b[i] + (math.pow(h/k, 2)*g(x[i+1], y[m]))
        b[lmax-n+1+i] = b[lmax-n+1+i] + (math.pow(h/k, 2)*g(x[i+1], y[0]))

    return b


# Define meshgrid boundary
a = 0
b = 2
c = 0
d = 1


# Define Dirichlet BC
def g(x,y):
    x = float(x)
    y = float(y)
    if x == 0.0:
        return 0.0
    elif y == 0.0:
        return x
    elif x == 2.0:
        return 2*math.exp(y)
    else:
        return x*math.exp(1)


# Define f(x)
f = lambda x, y: x*math.exp(y)
f = np.vectorize(f)

# Define number of points along each axis
n = 6
m = 5

# Calculate interval size
h = (b-a)/n
k = (d-c)/m

# Calculate A matrix
A = makeA(h, k, n, m)

# Create x and y vectors
x = np.linspace(a, b, num=n+1)
y = np.linspace(c, d, num=m+1)

# Calculate b vector
b = makeb(f, g, h, k, n, m, x, y)

# Find approximation
wl = np.linalg.solve(A,b)
print(wl)