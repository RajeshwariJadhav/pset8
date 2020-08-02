import numpy as np
import matplotlib.pyplot as plt

S = 3
w = 5
h = 4
M = 3

y = np.random.random((S, w*h)) #given
A = np.random.random((S, M)) #to optimize
Q = np.random.random((M, w*h)) #given
J = np.ones((S*w*h, S*M+M*w*h)) #jacobian
C = np.ones((S*w*h)) #cost

def cost(yrow, ycol):
    return (y[yrow, ycol] - np.dot(A[yrow, :], Q[:,ycol]))**2   

def formCost():
    for yrow in range(S):
        for ycol in range(w*h):
            C[yrow*w*h + ycol] = cost(yrow, ycol)
            #swh x 1
            
def Ajacobian(yrow, ycol, k):
    return (-2*y[yrow][ycol]*Q[k, ycol] + 2*np.dot(A[yrow,:], Q[:,ycol])*Q[k, ycol])

def Qjacobian(yrow, ycol, k):
    return (-2*y[yrow][ycol]*A[yrow, k] + 2*np.dot(A[yrow,:], Q[:,ycol])*A[yrow, k])

def formJacobian():
    
    for Jrow in range(S*w*h): #for each residual function
        yrow = Jrow // (w*h) #which y row residual
        ycol = Jrow % (w*h) #which y col residual
        
        #the A jacobian segment   
        for Arow in range(S): #each row in A
            for Acol in range(M): #each column in A
                Jcol = Arow * M + Acol 
                J[Jrow, Jcol] = Ajacobian(yrow, ycol, Acol)
                
        
        #the Q Jacobian segment
        for Qcol in range(w*h): #each row in Q
            for Qrow in range(M): #each column in Q
                Jcol = S*M + Qcol * M + Qrow #offset for A + index
                J[Jrow, Jcol] = Qjacobian(yrow, ycol, Qrow)
    

def GN(A, Q):
    maxIters = 12
    iterCount = 0
    cost = list()         
    iters = list()
    while (iterCount < maxIters):
        formJacobian()
        formCost()
        subtract = np.dot(np.dot(np.linalg.pinv(np.dot(J.T, J)), J.T), C)
        A = np.reshape(A, (S*M))
        Q = np.reshape(Q, (M*w*h), order = 'F')
        A = A - subtract[:S*M]
        Q = Q - subtract[S*M:]
        A = np.reshape(A, (S, M))
        Q = np.reshape(Q, (M, w*h), order = 'F')
        iters.append(iterCount)
        cost.append(np.sum(C))
        iterCount+=1
    print(cost)
    return A, Q


A, Q = GN(A, Q)

    
    