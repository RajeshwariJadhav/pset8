import numpy as np
import scipy.optimize as opt
#import opt_package.optimize_mine.py
import matplotlib.pyplot as plt
#exec("optimize_mine.py")
# Start with random A having independent columns, and set X accordingly
#np.random.seed(1)
# at every iteration:
# 1. h = minimize L1(x+dx) + Lambda (w(h)-r) --> 
# 2. add ax to A and find x accordingly

S = 4
M = 3
w = 4
h = 5

A = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,1,0]
])
# 4 x 3

y = np.array([
    [0, 1, 0, 6, 3, 0, 6, 0, 7, 8, 0, 0, 0, 3, 0, 1 , 0, 1, 3, 0],
    [0, 4, 7, 0, 0, 1, 2, 3, 6, 8, 5, 5, 8, 4, 7, 3 , 7, 0, 3, 4],
    [1, 5, 4, 3, 3, 0, 6, 0, 7, 3, 0, 1, 0, 3, 0, 1 , 4, 0, 0, 0],
    [1, 1, 2, 3, 4, 5, 6, 1, 7, 6, 8, 1, 9, 3, 5, 0 , 4, 0, 3, 2]
])
# 4 x 20

X = np.dot(np.linalg.pinv(A),y)
# 3 x 20

A = A.astype(float)
X = X.astype(float)


def psi(h, A, X):
    da = h[0]
    dx = h[1]
    ret = np.array([np.dot(A, dx) + np.dot(da, X), np.diag(np.dot(A.T, da))])
    return ret

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
    
def finalCost(y,A,X):
    return np.linalg.norm(y - np.dot(A, X))

def checkCost(X, dx):
    return np.linalg.norm(X + dx, 1)

def projection(A, X):
    for i in range(len(A[0])):
        A[:,i] = A[:,i]/np.linalg.norm(A[:,i])
    X = X + np.dot(np.linalg.pinv(A),(y-np.dot(A, X)))
    return A, X

def hCostOld(x, r, lambd, A, X):
    p = psi(x, A, X)
    p[0] = np.linalg.norm(p[0]-r[0])
    p[1] = np.linalg.norm(p[1]-r[1]) 
    return J(x, np.inf) + lambd*p


def hCost(x, r, lambd, A, Q):
    p = psi(x, A, Q)
    p[0] = np.linalg.norm(p[0]-r[0])
    p[1] = np.linalg.norm(p[1]-r[1])
    ret = J(x, np.inf) + np.linalg.norm(lambd*p)
    return ret

def findH(h, r, lambd):
    x0 = h.copy()
    #print("x0: ", x0)
    # print("r: ", r)
    # print("hCost: ", hCost)
    # print("x0: ", x0)
    # print("lambd: ", lambd)
    # print("A: ", A)
    # print("X: ", X)

    # print("r shape: ", np.shape(r))
    # print("x0.shape: ", np.shape(x0))
    # print("A.shape: ", np.shape(A))
    # print("X.shape: ", np.shape(X))

    #try:
    optimalH = opt.minimize(hCost, x0, args = (r, lambd, A, X), method = "CG")
    #    print("Optimal H: ", optimalH)
    #optimalH = minimize(hCost, x0, args = (r, lambd, A, X), method = "CG")
    #except Exception:
    #    import pdb; pdb.set_trace()
    #    raise
    
    return (optimalH.x)

def r(da, dx, hnew, h):
    r = r + psi(hnew)-psi(h)

    
def GN(A, X):
    #plotting h every loop
    all_cost = []
    
    maxIters = 200
    i = 0
    h = [np.random.random(A.shape), np.random.random(X.shape)]
    r = np.array([np.zeros(np.dot(A,X).shape),np.zeros(M)])
    lambd = 1
    iters = list()
    costs = list()
    while i < maxIters:
        #print("h_(da): ", h[0])
        all_cost.append(checkCost(X, h[1]))
        newh = findH(h, r, lambd)
        r = r + psi(newh, A, X)-psi(h, A, X)
        h = newh
        A += h[0]
        A, X = projection(A,X)
        A = np.nan_to_num(A)
        X = np.nan_to_num(X)
        i+=1
        iters.append(i)
        costs.append(finalCost(y,A,X))
    plt.plot(iters, costs)
    print("All cost: ", all_cost)
    plot2 = plt.figure(2)
    plt.plot(all_cost)
    plt.show()
    return A, X


r = np.zeros(M)
x0 = np.array([np.random.random(A.shape), np.random.random(X.shape)])

A, X = GN(A, X)
print("A ", A)
print("X", X)
print("AX", np.dot(A,X))
print("y", y)
