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
    Lcone = xyzbar[:,1]
    Mcone = xyzbar[:,2]
    Scone = xyzbar[:,3]
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
    cv.imshow("Sample Hyperspectral Image", np.reshape(reflectances[0], (255, 335)))
    cv.waitKey(0)
    cd.destroyAllWindows()
    plt.plot(reflectances)
    plt.show()
    illum = np.loadtxt('./data/illuminant_D65.csv',delimiter=',')
    illum = illum[20:81,:] #extracting 400-700 nm
    illum = illum[::2] #extracting every alternate row (data was originally spaced by 5 nm)
    illum = illum[:,1] #just the spectral profile

    # Possible Procedure: Display a hyperspectral image
    # Possible Procedure: Plot a graph of the reflectance spectrum at a pixel

    xyzbar = np.loadtxt('./data/CIE1931.csv',delimiter=',')
    xyzbar = xyzbar[::2]

    illuminate(reflectances, illum, xyzbar)

if __name__ == "__main__":
    main()
