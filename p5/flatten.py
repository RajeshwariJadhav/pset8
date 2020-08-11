import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random


def psi(h, A, X):
    da = h[0]
    dx = h[1]
    ret = np.array([np.dot(A, dx) + np.dot(da, X), np.diag(np.dot(A.T, da))])
    return ret

def r(da, dx, hnew, h):
    r = r + psi(hnew)-psi(h)

def J(h, nu, A, Q):
    da = h[0]
    dx = h[1]
    add = np.linalg.norm(dx)**2+np.linalg.norm(da)**2
    if (add <= nu):
        temp = float(np.linalg.norm(Q + dx, ord = 1))
        ret = np.random.random()/10000
        ret += temp
        return ret
    return np.inf 

def hCost(x, r, lambd, A, Q):
    #reshape -- unflatten
    div = A.shape[0]*A.shape[1]
    Apart = x[:div]
    Xpart = x[div:]
    AMat = np.reshape(Apart, ( A.shape[0],A.shape[1]))
    XMat = np.reshape(Xpart, (Q.shape[0], Q.shape[1]))
    x = [AMat, XMat]

    p = psi(x, A, Q)
    p[0] = np.linalg.norm(p[0]-r[0])
    p[1] = np.linalg.norm(p[1]-r[1])
    jval = J(x, np.inf, A, Q) 
    pval = np.linalg.norm(p[0]-r[0])+np.linalg.norm(p[1]-r[1])
    ret = jval + lambd*pval
    return ret

def findH(h, r, lambd, A, X): 
    #reshape -- flatten
    Aflat = np.reshape(h[0], (1, A.shape[0]*A.shape[1]))
    Xflat = np.reshape(h[1], (1, X.shape[0]*X.shape[1]))
    hVector = np.array(np.append(Aflat, Xflat))
    x0 = np.array(hVector) 
    
    optimalH = opt.minimize(hCost, x0, args = (r, lambd, A, X), method = "CG")
    print('optimal h found for update',optimalH.x)    
    return (optimalH.x)

def projection(A, X):
    #unit columns for A
    for i in range(len(A[0])):
        A[:,i] = A[:,i]/np.linalg.norm(A[:,i])
    #Frobenius norm for X    
    X = X + np.dot(np.linalg.pinv(A),(y-np.dot(A, X)))
    return A, X

def finalCost(h, r, lambd, A,X):
    ret = hCost(h, r, lambd, A, X)
    return ret
    
def GN(A, X):
    #initialize h to zeros
    h = [np.zeros(A.shape), np.zeros(X.shape)]
    r = np.array([np.zeros(np.dot(A,X).shape),np.zeros(M)])
    
    lambd = 0.01
    #cost of guessed A,X
    iters = [0]
    costs = [finalCost(np.zeros(72), r, lambd, A, X)]
    
    i = 0
    maxIters = 5
    while i < maxIters:
        #find the optimal h = [da, dx] that minimizes hCost
        newh = findH(h, r, lambd, A, X)
        cost = finalCost(newh,r,lambd,A,X)
        costs.append(cost)
        div = A.shape[0]*A.shape[1]
        Apart = newh[:div]
        Xpart = newh[div:]
        AMat = np.reshape(Apart, (A.shape[0],A.shape[1]))
        XMat = np.reshape(Xpart, (X.shape[0], X.shape[1]))
        newh = np.array([AMat, XMat])
        #update r
        r = r + psi(newh, A, X)-psi(h, A, X)
        #update h
        h = newh
        #update A and X according da and dx
        A += h[0]
        X += h[1]
        #project at every iter
        A, X = projection(A,X)
        iters.append(i+1)
        i+=1
    plt.plot(iters,costs)
    print("costs are", costs)
    return A, X

S = 4
M = 3
w = 4
h = 5
g = 0

A = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,1,0]
    ]) 

X = np.array([
    [0, 3, 0, 0, 0, 0, 6, 0, 7, 0, 0, 0, 0, 3, 0, 0 , 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0 , 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 , 0, 0, 0, 0]
])

y = np.dot(A,X)

A = np.zeros((S,M))
for i in range(S):
    col = i%M
    A[i, col] = 1

print(A)
A = A.astype(float)
X = np.dot(np.linalg.pinv(A),y) #initialize X according to A
X = X.astype(float)
Xinit = np.array(X)
Ainit = np.array(A)
print("A before GN", A.round(1))
print("X before GN", X.round(1))
print("AX before GN", np.dot(A,X).dot(1))
A, X = GN(A, X)
print("A after GN ", A.round(1))
print("X after GN", X.round(0))
print("AX after GN", np.dot(A,X).round(1))
print("y after GN", y)
g+=1
    
    
    
    
