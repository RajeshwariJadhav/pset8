import numpy as np
import cv2 as cv
from scipy.io import loadmat

Z = loadmat('./h_matrix.mat')
Z = np.transpose(Z["Z"], (2, 0, 1))
print("Z[0]:", Z)
cv.imshow("Z[0]: ", Z[0])
cv.waitKey(0)
cv.destroyAllWindows()
