import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random
import cv2 as cv
import os
from scipy.io import loadmat
# np.random.seed(0)

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
    jval = J(x, np.inf) 
    pval = np.sqrt(np.linalg.norm(p[0]-r[0])**2+np.linalg.norm(p[1]-r[1])**2)
    ret = jval + lambd*pval
    return ret

def findH(h, r, lambd, A, X):
    x0 = h
    optimalH = opt.minimize(hCost, x0, args = (r, lambd, A, X), method = "CG")
    return (optimalH.x)

def projection(A, X):
    #unit columns for A
    Ainit = np.array(A)
    Xinit = np.array(X)
    for i in range(len(A[0])):
        A[:,i] = A[:,i]/np.linalg.norm(A[:,i])
    #Frobenius norm for X    
    X = X + np.dot(np.linalg.pinv(A),(y-np.dot(A, X)))
    return A, X

def finalCost(h, r, lambd, A,X,dx, y):
    ret = hCost(h, r, lambd, A, X)
    # ret = np.linalg.norm(y-np.dot(A,X))
    return ret
    
def GN(A, X):
    #initialize h randomly
    AX = np.dot(A,X)
    im = np.reshape(AX, (images.shape))
    cv.imshow("old slice of ax ", im[0,:,:])

    h = [np.random.random(A.shape), np.random.random(X.shape)]
    r = np.array([np.zeros(np.dot(A,X).shape),np.zeros(M)])
    costs = list()
    iters = list()
    lambd = 0.001
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
        iters.append(i)
        costs.append(finalCost(h,r,lambd,A,X,h[1], y))
        i+=1
    plt.plot(iters,costs)
    print("these are costs", costs)
    return A, X



def main_manchester_data():
    ref4_dict = loadmat("../p3/manchester/assets/ref4_scene4.mat")
    reflectances = ref4_dict['reflectances']
    reflectances = np.transpose(reflectances, (2, 0, 1))
    original_hires = reflectances
    reflectances = reflectances[::, ::4, ::4]
    rgb = cv.imread("../p3/manchester/manchester_rgb.jpg", 1)
    cv.imshow("rgb", rgb)

    print("rgb in man func:", rgb.shape)
    print("img in man func: ", reflectances.shape)
    return original_hires, rgb, reflectances


original_hires, rgb, images = main_manchester_data()
print("RGB shape: ", rgb.shape)
print("images shape: ", images.shape)
original_lowres = images[0]

y = np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2]))
S = y.shape[0]
M = 3
w = images.shape[1]
h = images.shape[2]
A = np.zeros((S,M))
#initialize A to have independent cols -- random entries
for i in range(S):
    col = i%M
    A[i, col] = np.random.random()

#initialize X randomly as well
X = np.random.random((M,w*h))
A, X = GN(A, X)

AX = np.dot(A,X)
im = np.reshape(AX, (images.shape))
cv.imshow("slice of ax ", im[0,:,:])

print("X = ",X)

print("A = ", A)

# #********************FINDING H *********************

# Manchester yRGB
# yRGB = rgb
# yRGB = cv.normalize(yRGB, None, np.amin(y), np.amax(y), cv.NORM_MINMAX, dtype=cv.CV_32F)
# yRGB = np.reshape(yRGB, (yRGB.shape[2], yRGB.shape[0]*yRGB.shape[1]))

# def minHCost(x, yIJ, pRGB, A, epsilon):
#     #here x is H (to be found)
#     l1 = np.linalg.norm(x, ord=1)
#     l2 = np.linalg.norm(yIJ-np.dot(pRGB,np.dot(A,x)))
#     return l1+2*l2
    
# def findFinalH(h,yRGB, pRGB, A):
#     x0 = h
#     epsilon = 5
#     optimalH = opt.minimize(minHCost, x0, args = (yRGB, pRGB, A, epsilon), method = "CG")
#     return (optimalH.x)

# #pRGB = np.ones((3,S))
# pRGB = np.loadtxt('../p3/data/CIE1931.csv', delimiter=',')[::2]
# pRGB = pRGB[::3]
# pRGB = np.transpose(pRGB)[1:4]
# print("pRGB shape: ", pRGB.shape)
# print(pRGB)
# # so far initial guess works the best but optimization function might need more work.
# #is pinv good enough?
# H = np.dot(np.linalg.pinv(A), np.dot(np.linalg.pinv(pRGB), yRGB)) 
# print("Old H: ", H)
# # for c in range(yRGB.shape[1]):
# #    yIJ = yRGB[:,c]
# #    x =np.array(H[:,c])
# #    a = findFinalH(x, yIJ, pRGB, A)
# #    H[:,c]=a
# # print("New H: ", H)
# #*************************FINDING Z****************

# Z = np.dot(A, H)
# print(Z.shape) #yay

# Z = np.reshape(Z, (Z.shape[0], h_l,w_l))
# print(Z.shape)
# for i in range(Z.shape[0]):
#     Z[i] = cv.normalize(Z[i],  None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)

# print("Max in final Z: ", np.amax(Z))
# print("Min in final Z: ", np.amin(Z))
# #print("Final Z print: ", Z)
# cv.imshow("Final Z", Z[0])
# cv.imshow("original low res Z[0]", original_lowres)
# cv.imshow("original hi res Z[0]", original_hires)
# plt.imshow(Z[0,:,:], cmap = 'gray')
# plt.show()
cv.waitKey(0)
