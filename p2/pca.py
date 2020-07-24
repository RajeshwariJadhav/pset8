import numpy as np
import cv2 as cv
from scipy.io import loadmat

# Extract covariance matrix
def extract_cov(A):
    A_flat = np.reshape(A, (A.shape[0]*A.shape[1], A.shape[2]))
    # Calculate mean spectrum
    print("8")
    avg_spec = np.mean(A_flat, axis=0)
    print("avg_spec shape", avg_spec.shape)
    B = np.array([vect - avg_spec for vect in A_flat])
    print("11")
    print("Shape of B", B.shape)
    C = np.matmul(B.T, B)
    print("C shape", C.shape)
    return B, C, avg_spec

def extract_pc(C):
    eigVals, eigVecs = np.linalg.eig(C)
    idx = eigVals.argsort()
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:,idx]
    return eigVecs

def data_projection(A, eigVecs, B, avg_spec, r):
    R = np.array([eigVecs[:,i] for i in range(r)]).T
    print("R shape", R.shape)
    M = np.mean(A, axis=2)
    R_new = np.matmul(R, R.T)
    print("R new shape", R_new.shape)
    # note that this was a mistake in the textbook. It should be B.T not B
    A_new = np.matmul(R_new, B.T)
    A_new = np.array([vect + avg_spec for vect in A_new.T]).T
    return A_new

def main():
    dir_img = "./test_data/nature.jpg"
    img = cv.imread(dir_img, 1)

    # annots = loadmat('./test_data/Indian_pines.mat')
    # img = annots['indian_pines']
    
    B, C, avg_spec = extract_cov(img)
    eigVecs = extract_pc(C)
    A_new = data_projection(img, eigVecs, B, avg_spec, 20)
    pc1 = A_new[0]
    pc2 = A_new[1]
    pc3 = A_new[2]
    pc1 = np.reshape(np.uint8(pc1), (img.shape[0], img.shape[1]))
    pc2 = np.reshape(np.uint8(pc2), (img.shape[0], img.shape[1]))
    pc3 = np.reshape(np.uint8(pc3), (img.shape[0], img.shape[1]))
    img_new = cv.merge([pc1,pc2,pc3])
    # img_new = pc1
    cv.imshow("img_new", img_new)
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()