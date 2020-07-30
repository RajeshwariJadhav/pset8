import numpy as np
import cv2 as cv
from scipy.io import loadmat
import os
# Extract covariance matrix

def load_images_from_folder(folder):
    i = 0
    images = np.zeros((31,512,512,3))
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images[i]=img
            i+=1
    return images

def extract_cov(A):
    A_flat = np.reshape(A, (A.shape[0]*A.shape[1], A.shape[2]))
    # Calculate mean spectrum
    avg_spec = np.mean(A_flat, axis=0)
    B = np.array([vect - avg_spec for vect in A_flat])
    C = np.matmul(B.T, B)
    return B, C, avg_spec

def extract_pc(C):
    eigVals, eigVecs = np.linalg.eig(C)
    idx = eigVals.argsort()
    eigVals = np.flip(eigVals[idx])
    
    eigVecs = np.flip(eigVecs[:,idx])
    return eigVecs

def data_projection(A, eigVecs, B, avg_spec, r):
    R = np.array([eigVecs[:,i] for i in range(r)]).T
    M = np.mean(A, axis=2)
    R_new = np.matmul(R, R.T)
    # note that this was a mistake in the textbook. It should be B.T not B
    A_new = np.matmul(R_new, B.T)
    A_new = np.array([vect + avg_spec for vect in A_new.T]).T
    return A_new

def main():
    #dir_img = "./test_data/nature.jpg"
    #img = cv.imread(dir_img, 1)
    pictures = load_images_from_folder('../p3//data/thread_spools_ms')
    img = np.uint8(pictures[:,:,:,1]) #data is originally in duplicated triples
    dim = img.shape
    cv.imshow("orig hyperspectral image", img[1])
    img = np.transpose(img, (1,2,0))
    # max_spectra = np.amax(img, axis=(0,1))
    print("img shape: ", img.shape)

    #normalizing
    # for i in range(img.shape[2]):
    #     img[:,:,i] = np.divide(img[:,:,i],  max_spectra[i])

    B, C, avg_spec = extract_cov(img)
    eigVecs = extract_pc(C)
    A_out = np.array(data_projection(img, eigVecs, B, avg_spec, 2)).real
    A_reshaped = np.zeros(dim)
    for i in range(A_reshaped.shape[0]):
        A_reshaped[i] = np.reshape(A_out[i], (dim[1], dim[2]))
        A_reshaped[i] = cv.normalize(A_reshaped[i], None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    cv.imshow("img_new_pc1.1", A_reshaped[0])
    cv.imshow("img_new_pc1.2", A_reshaped[1])
    cv.imshow("img_new_pc1.3", A_reshaped[2])

    #cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #pc1 = np.reshape(np.uint8(pc1), (img.shape[0], img.shape[1]))
    #pc2 = np.reshape(np.uint8(pc2), (img.shape[0], img.shape[1]))
    #pc3 = np.reshape(np.uint8(pc3), (img.shape[0], img.shape[1]))
    # pc1 = np.reshape(pc1/255, (img.shape[0], img.shape[1]))
    # pc2 = np.reshape(pc2/255, (img.shape[0], img.shape[1]))
    # pc3 = np.reshape(pc3/255, (img.shape[0], img.shape[1]))
    # img_new = cv.merge([pc1,pc2,pc3])
    # img_new = pc1
    
if __name__ == "__main__":
    main()
