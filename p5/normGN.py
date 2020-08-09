import numpy as np
import scipy.optimize as opt
#import opt_package.optimize_mine.py
import matplotlib.pyplot as plt
import random


def psi(h, A, X):
    da = h[0]
    dx = h[1]
    ret = np.array([np.dot(A, dx) + np.dot(da, X), np.diag(np.dot(A.T, da))])
    return ret

def r(da, dx, hnew, h):
    r = r + psi(hnew)-psi(h)

def J(h, nu):
    da = h[0]
    dx = h[1]
    add = np.linalg.norm(dx)**2+np.linalg.norm(da)**2
    if (add <= nu):
        temp = float(np.linalg.norm(X + dx, ord = 1))
        ret = np.random.random()/10000
        ret += temp
        return ret
    return np.inf 

def hCost(x, r, lambd, A, Q):
    p = psi(x, A, Q)
    p[0] = np.linalg.norm(p[0]-r[0])
    p[1] = np.linalg.norm(p[1]-r[1])
    jval = J(x, np.inf) 
    pval = np.sqrt(np.linalg.norm(p[0]-r[0])**2+np.linalg.norm(p[1]-r[1])**2)
    ret = jval + lambd*pval
    return ret

def findH(h, r, lambd, A, X):
    x0 = h
    optimalH = opt.minimize(hCost, x0, args = (r, lambd, A, X), method = "CG")
    #    print("Optimal H: ", optimalH)
    #optimalH = minimize(hCost, x0, args = (r, lambd, A, X), method = "CG")
    #except Exception:
    #    import pdb; pdb.set_trace()
    #    raise
    
    return (optimalH.x)

def projection(A, X):
    #unit columns for A
    for i in range(len(A[0])):
        A[:,i] = A[:,i]/np.linalg.norm(A[:,i])
    #Frobenius norm for X    
    X = X + np.dot(np.linalg.pinv(A),(y-np.dot(A, X)))
    return A, X

def finalCost(h, r, lambd, A,X,dx, y):
    ret = hCost(h, r, lambd, A, X)
    return ret
    
def GN(A, X):
    #initialize h to zeros
    h = [np.zeros(A.shape), np.zeros(X.shape)]
    r = np.array([np.zeros(np.dot(A,X).shape),np.zeros(M)])
    
    lambd = 0.01
    iters = list()
    costs = list()
    
    i = 0
    maxIters = 100
    while i < maxIters:
        #find the optimal h = [da, dx] that minimizes hCost
        newh = findH(h, r, lambd, A, X)
        #update r
        r = r + psi(newh, A, X)-psi(h, A, X)
        #update h
        h = newh
        #update A and X according da and dx
        A += h[0]
        X += h[1]
        #project at every iter
        A, X = projection(A,X)
        i+=1
    return A, X

def cost_recon(yrow, ycol):
    return (y[yrow, ycol] - np.dot(A[yrow, :], Q[:,ycol]))**2   

def formCost_recon():
    for yrow in range(S):
        for ycol in range(w*h):
            C[yrow*w*h + ycol] = cost(yrow, ycol)
            #swh x 1

def hjacobian(Y_rgb, A, h, yrow, ycol, k):
    return (-2*Y_rgb[yrow][ycol]*A[k, ycol] + 2*np.dot(h[yrow,:], A[:,ycol])*A[k, ycol])

#def Qjacobian(yrow, ycol, k):
#    return (-2*y[yrow][ycol]*A[yrow, k] + 2*np.dot(A[yrow,:], Q[:,ycol])*A[yrow, k])

def formJacobian_recon(Y_rgb, P_rgb, A, h):

    for Jrow in range(S*w*h): #for each residual function
        yrow = Jrow // (w*h) #which y row residual
        ycol = Jrow % (w*h) #which y col residual

        #the A jacobian segment   
        for hrow in range(S): #each row in A
            for hcol in range(M): #each column in A
                Jcol = hrow * M + hcol 
                J[Jrow, Jcol] = hjacobian(Y_rgb, A, h, yrow, ycol, hcol)
        return J

def GN_recon(Y_rgb, P_rgb, A, h):
    maxIters = 12
    iterCount = 0
    cost = list()         
    iters = list()
    while (iterCount < maxIters):
        J = formJacobian(Y_rgb, P_rgb, A, h)
        formCost()
        subtract = np.dot(np.dot(np.linalg.pinv(np.dot(J.T, J)), J.T), C)
        h = np.reshape(h, (S*M))
        Q = np.reshape(Q, (M*w*h), order = 'F')
        h = h - subtract[:S*M]
        Q = Q - subtract[S*M:]
        h = np.reshape(h, (S, M))
        Q = np.reshape(Q, (M, w*h), order = 'F')
        iters.append(iterCount)
        cost.append(np.sum(C))
        iterCount+=1
    print(cost)
    return A, Q
    
S = 4
M = 3
w = 4
h = 5
g = 0


P_rgb = np.array([])

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

while g<2:
    A = np.array([
    [0,0,1],
    [1,0,0],
    [0,1,0],
    [0,1,0]
    ])    #different from the A used to find y

    A = np.array([
    [0.245,0,0],
    [0,0.874,0],
    [0,0,0.598],
    [0,0,0]
    ])    
        
    X = np.dot(np.linalg.pinv(A),y) #initialize X according to A
    X = X.astype(float)
    A, X = GN(A, X)
    print(np.linalg.norm(y-np.dot(A,X)))
    print("A ", A)
    print("X", X.round(1))
    print("AX", np.dot(A,X).round(3))
    print("y", y)
    g+=1

