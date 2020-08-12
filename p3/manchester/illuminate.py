from scipy.io import loadmat
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

def manchester_post_processing(RGB):

    z = max(RGB[244][17])
    RGB_clip = np.where(RGB>z, z, RGB)

    RGB_clip = RGB_clip/z
    
    RGB = RGB_clip**0.4
    print("RGB max: ", np.amax(RGB))
    RGB_int = np.uint8(RGB*255)
    #RGB_int =  cv.normalize(RGB, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

    return RGB_int

def switch_randb(RGB):
    temp_rgb = np.transpose(RGB, (2, 0, 1))
    temp = temp_rgb[0].copy()
    temp_rgb[0] = temp_rgb[2].copy()
    temp_rgb[2] = temp
    RGB = np.transpose(temp_rgb, (1,2,0))
    return RGB
    
def load_images_from_folder(folder):
    i = 0
    images = np.zeros((31,512,512,3)) 
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images[i]=img
            i+=1
    return images

def XYZ_to_sRGB(XYZ):
    dim = XYZ.shape
    XYZ = np.transpose(XYZ, (1,0,2))
    XYZ = np.reshape(XYZ, (dim[0]*dim[1], dim[2]))
    # Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in
    # IEC_61966-2-1.pdf).
    M = [[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0414,],
         [0.0557, -0.2040, 1.0570]]
    sRGB = np.matmul(M, XYZ.T).T
    sRGB = np.reshape(sRGB, (dim[1], dim[0], dim[2]))
    sRGB = np.transpose(sRGB, (1, 0, 2))
    return sRGB

def illuminate(reflectances, illum, xyzbar):
    reflectances = reflectances/np.max(reflectances)
    dim_refl = reflectances.shape
    radiances = np.zeros(dim_refl)
    for i in range(dim_refl[2]):
        radiances[:,:,i] = reflectances[:,:,i]*illum[i]
    # Possible Procedure: Plot the radiances as well
    # slice_rad = radiances[141][75]
    # plt.plot(range(400, 730, 10), slice_rad)
    # plt.show()
    radiances = np.transpose(radiances, (1,0,2))
    radiances = np.reshape(radiances, (dim_refl[0]*dim_refl[1], dim_refl[2]))

    # Note that they use matrix multiplication here, instead of regular mult + integration
    # like we did
    XYZ = np.matmul(xyzbar.T, radiances.T).T

    XYZ = np.reshape(XYZ, (dim_refl[1], dim_refl[0], 3))
    XYZ = np.transpose(XYZ, (1,0, 2))

    # Normalize XYZ vals
    # Note that maybe you should normalize by the color, individually
    XYZ = XYZ/np.max(XYZ)
    RGB = XYZ_to_sRGB(XYZ)
    RGB = np.where(RGB<0, 0, RGB)
    RGB = np.where(RGB>1, 1, RGB)

    RGB = switch_randb(RGB)

    RGB = manchester_post_processing(RGB)

    cv.imshow("Out", RGB)
    cv.imshow("Out", RGB**0.4)
    cv.imwrite("manchester_rgb.jpg", RGB)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def main_manchester_data():
    ref4_dict = loadmat("../manchester/assets/ref4_scene4.mat")
    reflectances = ref4_dict['reflectances']
    # Possible Procedure: Display a hyperspectral image
    # Possible Procedure: Plot a graph of the reflectance spectrum at a pixel

    illum_dict = loadmat('../manchester/assets/illum_6500.mat')
    illum = illum_dict['illum_6500']

    xyzbar_dict = loadmat("../manchester/assets/xyzbar.mat")
    xyzbar = xyzbar_dict['xyzbar']
    illuminate(reflectances, illum, xyzbar)

def main():
    illum = np.loadtxt('../data/illuminant_D65.csv',delimiter=',')
    illum = illum[20:81,:] #extracting 400-700 nm
    illum = illum[::2] #extracting every alternate row (data was originally spaced by 5 nm)
    illumSpec = illum[:,1] #just the spectral profile
    print("illumspec shape", illumSpec.shape)

    pictures = load_images_from_folder('../data/thread_spools_ms') #get all images
    images = pictures[:,:,:,1] #data is originally in duplicated triples
    images = np.transpose(images, (1, 2, 0))
    print("images shape: ", images.shape)
    
    xyzbar_dict = loadmat("../manchester/assets/xyzbar.mat")
    xyzbar = xyzbar_dict['xyzbar']
    xyzbar = xyzbar[0:31]
    illuminate(images, illumSpec, xyzbar)

if __name__ == "__main__":
    main_manchester_data()
    
