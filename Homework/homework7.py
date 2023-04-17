# Import required packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

# PROBLEM 1
# Define N
Ns = [5,35]
for N in Ns:
    # Define A matrix
    row = np.zeros(N)
    val = 2
    for j in np.linspace(1, N, num=int((N+1)/2)):
        row[int(j-1)] = val
        val = val * -1
    A = np.zeros((N, N))
    neg1 = 1
    for j in np.linspace(1, N, num=int((N+1)/2)):
        A[int(j-1), :] = row*neg1
        neg1 = neg1*-1
    for j in np.arange(N):
        A[j,j] = math.pow((j+1)*math.pi,2)
        if ((j+1) % 2) == 0:
            pass
        else:
            A[j,j] = A[j,j] + 2

    # Create b vector
    b = np.zeros(N)
    for j in np.arange(N):
        if ((j+1) % 2) == 0:
            pass
        else:
            b[j] = 1

    coef = np.linalg.solve(A, b)

    # Define basis func
    phi = lambda k, x: math.sqrt(2)*math.sin(k*math.pi*x)
    phi = np.vectorize(phi)

    # Define x vector
    x = np.linspace(0,1)

    # Calculate w
    w = np.zeros(len(x))
    for i in np.arange(N):
        w = w + coef[i]*phi(i+1, x)

    plt.plot(x,w)

plt.xlabel('x')
plt.ylabel('w')
plt.title('Approx for Different Ns')
plt.legend(['N=5','N=35'])
plt.grid(True)
plt.show()

# PROBLEM 2
# Define N
N = 32

# Calculate h
h = 1/(N+1)

# Create K matrix
diag1 = -1*np.ones(N+2)*(1/h)
diag1[0] = diag1[0]-1
diag1[-1] = diag1[-1]-1
diag2 = np.ones(N+2)*(1/h)
K = scipy.sparse.spdiags(np.array([diag2,diag1,diag2]),[-1,0,1]).toarray()
K[0,1] = K[0,1]*2
K[1,0] = K[1,0]*2
K[-2,-1] = K[-2,-1]*2
K[-1,-2] = K[-1,-2]*2

# Create M matrix
diag1 = np.ones(N+2)*2*h/3
diag1[0] = diag1[0]/2
diag1[-1] = diag1[-1]/2
diag2 = np.ones(N+2)*7*h/6
M = scipy.sparse.spdiags(np.array([diag2,diag1,diag2]),[-1,0,1]).toarray()
M[0,1] = 2*h/3
M[1,0] = 2*h/3
M[-2,-1] = 2*h/3
M[-1,-2] = 2*h/3

# Create b vector
b = np.ones(N+2)*h
b[0] = b[0]/2
b[-1] = b[-1]/2

# Define hat function
def hat(k,kmax,xk,h,x):
    phik = np.zeros(len(x))

    for i in np.arange(len(x)):
        if (x[i] >= xk[k-1]) & (x[i] < xk[k]) & (k != 0):
            phik[i] = (x[i]-xk[k-1])/h
        elif (x[i] < xk[k+1]) & (k != kmax):
            phik[i] = (xk[k+1]-x[i])/h
    return phik



es = [0.1,0.25,1.0]
for e in es:
    # Create A matrix
    A = (e*K)+M

    # Solve for coefficients
    coef = np.linalg.solve(A,b)

    # Create x vector
    x = np.linspace(0,1)

    # Create approximation
    w = np.zeros(len(x))
    for i in np.arange(N+1):
        w = w+(coef[i]*hat(i,N+1,np.linspace(0,1,num=N+2),h,x))

    plt.plot(x,w)

plt.xlabel('x')
plt.ylabel('w')
plt.title('Approx for Different e')
plt.legend(['e=0.1','e=0.25','e=1'])
plt.grid(True)
plt.show()