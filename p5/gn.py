import numpy as np
import matplotlib.pyplot as plt

S = 3
w = 5
h = 4
M = 3
alpha = 5*10**(-6) #controls the rate for minimizing Q's L1 norm

y = np.random.random((S, w*h)) #given

J = np.zeros((S*w*h, S*M+M*w*h)) #jacobian
C = np.ones((S*w*h)) #cost

def cost(yrow, ycol, A, Q):
    return (((y[yrow, ycol] - np.dot(A[yrow, :], Q[:,ycol]))**2)+alpha*np.linalg.norm(Q, ord= 1))

def formCost(A,Q):
    for yrow in range(S):
        for ycol in range(w*h):
            C[yrow*w*h + ycol] = cost(yrow, ycol, A, Q)
            #swh x 1
            
def Ajacobian(yrow, ycol, k, A, Q):
    return (-2*y[yrow][ycol]*Q[k, ycol] + 2*np.dot(A[yrow,:], Q[:,ycol])*Q[k, ycol])

def formJacobian(A,Q):

    for Jrow in range(S*w*h): #for each residual function
        yrow = Jrow // (w*h) #which y row residual
        ycol = Jrow % (w*h) #which y col residual
        
        #the A jacobian segment   
        for Arow in range(S): #each row in A
            for Acol in range(M): #each column in A
                Jcol = Arow * M + Acol 
                J[Jrow, Jcol] = Ajacobian(yrow, ycol, Acol, A, Q)
                
        
        #the Q Jacobian segment
        for Qcol in range(w*h): #each row in Q
            for Qrow in range(M): #each column in Q
                Jcol = S*M + Qcol * M + Qrow #offset for A + index
                J[Jrow, Jcol] = Qjacobian(yrow, ycol, Qrow, A, Q)
    
    
A = np.random.random((S, M)) #to optimize
Q = np.random.random((M, w*h)) #given
def GN(A, Q):
    maxIters = 100
    iterCount = 0
    cost = list()         
    iters = list()
    while (iterCount < maxIters):
  
        formJacobian(A,Q)
        formCost(A,Q)
        
        #update A
        subtract = np.dot(np.linalg.pinv(J[:,:S*M]), C)
        A = np.reshape(A, (S*M))
        A = A - subtract
        A = np.reshape(A, (S, M))
        
        #update Q
        subtract = np.dot(np.linalg.pinv(J[:,S*M:]), C)
        Q = np.reshape(Q, (M*w*h), order = 'F')
        Q = Q - subtract
        Q = np.reshape(Q, (M, w*h), order = 'F')
        
        iters.append(iterCount)
        cost.append(np.sum(C))
        iterCount+=1
    print(cost)
    plt.plot(iters, cost)
    return A, Q


A, Q = GN(A, Q)

    
