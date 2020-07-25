import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import scipy.integrate

illum = np.loadtxt('./data/illuminant_D65.csv',delimiter=',')
illum = illum[20:81,:] #extracting 400-700 nm
illum = illum[::2] #extracting every alternate row (data was originally spaced by 5 nm)
illumSpec = illum[:,1] #just the spectral profile
print("illumspec shape", illumSpec.shape)

def load_images_from_folder(folder):
    i = 0
    images = np.zeros((31,512,512,3)) 
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images[i]=img
            i+=1
    return images

pictures = load_images_from_folder('./data/thread_spools_ms') #get all images
images = pictures[:,:,:,1] #data is originally in duplicated triples

radiance = np.zeros((31,512,512)) 
count = 0
for i in range(512):
    for j in range(512):
        radiance[:,i,j] = np.multiply(images[:,i,j],illumSpec) #l = r*e (elementwise multiplication)

sensitivity = np.loadtxt('./data/coneSensitivity.csv',delimiter=',')
sensitivity = sensitivity[2:63:2] #extracting 400-700 nm with 10 step.
#separating cone data
Lcone = sensitivity[:,1]
Mcone = sensitivity[:,2]
Scone = sensitivity[:,3]
X = sensitivity[:,0]

#making the L-cone component
lconeimg = np.zeros((512,512))
for i in range(512):
    for j in range(512):
        inside = np.multiply(radiance[:,i,j],Lcone) #s*l --> elementwise mult.
        lconeimg[i,j] = scipy.integrate.simps(inside, X) #integration
        
#making the M-cone component
mconeimg = np.zeros((512,512))
for i in range(512):
    for j in range(512):
        inside = np.multiply(radiance[:,i,j],Mcone)
        mconeimg[i,j] = scipy.integrate.simps(inside, X)

#making the S-cone component
sconeimg = np.zeros((512,512))
for i in range(512):
    for j in range(512):
        inside = np.multiply(radiance[:,i,j],Scone)
        sconeimg[i,j] = scipy.integrate.simps(inside, X)

#rescaling images to have 0-1 range
print("l max and min: ", np.amax(lconeimg), ", ", np.amin(lconeimg))
print("m max and min: ", np.amax(mconeimg), ", ", np.amin(mconeimg))
print("s max and min: ", np.amax(sconeimg), ", ", np.amin(sconeimg))

lconeimg/=np.amax(lconeimg)
mconeimg/=np.amax(mconeimg)
sconeimg/=np.amax(sconeimg)

lconeimg = np.uint8(lconeimg*255)
mconeimg = np.uint8(mconeimg*255)
sconeimg = np.uint8(sconeimg*255)

print("l mean: ", np.mean(lconeimg))
print("m mean: ", np.mean(mconeimg))
print("s mean: ", np.mean(sconeimg))

# merging RGB
img = cv2.merge((lconeimg, mconeimg, sconeimg))
img_real = cv2.imread("./data/thread_spools_RGB.bmp")
cv2.imshow("Real Image", img_real)
cv2.imshow("Constructed Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

