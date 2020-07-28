from scipy.io import loadmat
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import skimage.color as sc
import os
import scipy.integrate


def illuminate(reflectances, illum, xyzbar):
    reflectances = reflectances/np.max(reflectances)
    dim_refl = reflectances.shape
    radiances = np.zeros(dim_refl)

    for i in range(dim_refl[2]):
        radiances[:,:,i] = reflectances[:,:,i]*illum[i]

    radiance = radiances
    Lcone = xyzbar[:,0]
    Mcone = xyzbar[:,1]
    Scone = xyzbar[:,2]
    X = np.arange(400,710,10)

    #making the L-cone component
    lconeimg = np.zeros((255,335))
    for i in range(255):
        for j in range(335):
            inside = np.multiply(radiance[i,j,:],Lcone) #s*l --> elementwise mult
            lconeimg[i,j] = scipy.integrate.simps(inside, X) #integration
            
    #making the M-cone component
    mconeimg = np.zeros((255,335))
    for i in range(255):
        for j in range(335):
            inside = np.multiply(radiance[i,j,:],Mcone) #s*l --> elementwise mult
            mconeimg[i,j] = scipy.integrate.simps(inside, X) #integration
    
    #making the S-cone component
    sconeimg = np.zeros((255,335))
    for i in range(255):
        for j in range(335):
            inside = np.multiply(radiance[i,j,:],Scone) #s*l --> elementwise mult
            sconeimg[i,j] = scipy.integrate.simps(inside, X) #integration

    # merging RGB
    XYZ = cv.merge((lconeimg, mconeimg, sconeimg))

    XYZ = XYZ/np.max(XYZ)
    RGB = sc.xyz2rgb(XYZ)
    RGB = np.where(RGB<0, 0, RGB)
    RGB = np.where(RGB>1, 1, RGB)
    temp_rgb = np.transpose(RGB, (2, 0, 1))
    temp = temp_rgb[0].copy()
    temp_rgb[0] = temp_rgb[2].copy()
    temp_rgb[2] = temp
    RGB = np.transpose(temp_rgb, (1,2,0))
    RGB = RGB**0.4

    cv.imshow("rgb", RGB)
    cv.imshow("xyz", XYZ)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    ref4_dict = loadmat("./manchester/assets/ref4_scene4.mat")
    reflectances = ref4_dict['reflectances']
    reflectances = np.array(reflectances[:,:,:31])

    # Possible Procedure: Display a hyperspectral image
    # Possible Procedure: Plot a graph of the reflectance spectrum at a pixel
    illum_dict = loadmat('./manchester/assets/illum_25000.mat')
    illum = illum_dict['illum_25000']
    illum = np.array(illum[:31])

    xyzbar_dict = loadmat("./manchester/assets/xyzbar.mat")
    xyzbar = xyzbar_dict['xyzbar']
    xyzbar = np.array(xyzbar[:31])

    illuminate(reflectances, illum, xyzbar)

if __name__ == "__main__":
    main()
