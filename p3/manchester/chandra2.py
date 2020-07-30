from scipy.io import loadmat
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import skimage.color as sc


def XYZ_to_sRGB(XYZ):
    dim = XYZ.shape
    print("dim inside XYZ_to_RGB", dim)
    XYZ = np.transpose(XYZ, (1,0,2))
    XYZ = np.reshape(XYZ, (dim[0]*dim[1], dim[2]))
    print("some vals in XYZ after resha within func: ", XYZ[0][0], ", ", XYZ[1][0], ", ", XYZ[0][1])
    # Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in
    # IEC_61966-2-1.pdf).
    M = [[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0414,],
         [0.0557, -0.2040, 1.0570]]
    sRGB = np.matmul(M, XYZ.T).T
    print("some vals in sRGB within func: ", sRGB[0][0], ", ", sRGB[1][0], ", ", sRGB[0][1])
    sRGB = np.reshape(sRGB, (dim[1], dim[0], dim[2]))
    sRGB = np.transpose(sRGB, (1, 0, 2))
    print("some vals in sRGB within func after resha: ", sRGB[0][0][0], ", ", sRGB[1][0][0], ", ", sRGB[0][1][0])
    return sRGB

def illuminate(reflectances, illum, xyzbar):
    reflectances = reflectances/np.max(reflectances)
    dim_refl = reflectances.shape
    radiances = np.zeros(dim_refl)
    print("some vals in original reflectance and illum matricies: ", reflectances[0][0][0], ", ", reflectances[1][0][0], ", ",  illum[1])
    for i in range(dim_refl[2]):
        radiances[:,:,i] = reflectances[:,:,i]*illum[i]
    print("some vals in original radiances: ", radiances[0][0][0], ", ", radiances[1][0][0])
    # Possible Procedure: Plot the radiances as well
    # slice_rad = radiances[141][75]
    # plt.plot(range(400, 730, 10), slice_rad)
    # plt.show()
    # print("radiances shape before reshape: ", radiances.shape)

    print("Dim_refl: ", dim_refl)
    radiances = np.transpose(radiances, (1,0,2))
    radiances = np.reshape(radiances, (dim_refl[0]*dim_refl[1], dim_refl[2]))

    # Note that they use matrix multiplication here, instead of regular mult + integration
    # like we did
    print("xyzbar shape: ", xyzbar.shape, " radiances.shape: ", radiances.shape)
    print("some vals in xyzbar and radiances: ", radiances[0][0], ", ", radiances[1][0], ", ", radiances[0][1])
    XYZ = np.matmul(xyzbar.T, radiances.T).T
    print("XYZ shape: ", XYZ.shape)
    print("some vals in XYZ: ", XYZ[0][0], ", ", XYZ[1][0], ", ", XYZ[0][1])

    XYZ = np.reshape(XYZ, (dim_refl[1], dim_refl[0], 3))
    XYZ = np.transpose(XYZ, (1,0, 2))
    print("some vals in XYZ after 3d: ", XYZ[0][0][0], ", ", XYZ[1][0][0], ", ", XYZ[0][1][0])
    # Normalize XYZ vals
    # Note that maybe you should normalize by the color, individually
    XYZ = XYZ/np.max(XYZ)
    print("some vals in XYZ after norm: ", XYZ[0][0][0], ", ", XYZ[1][0][0], ", ", XYZ[0][1][0])
    # RGB = XYZ_to_sRGB(XYZ)
    RGB = sc.xyz2rgb(XYZ)
    cv.imshow("origRGBH", RGB)
    print("some vals in RGB: ", RGB[0][0][0], ", ", RGB[1][0][0], ", ", RGB[0][1][0])
    RGB = np.where(RGB<0, 0, RGB)
    RGB = np.where(RGB>1, 1, RGB)
    print("some vals in RGB after getting rid of outliers: ", RGB[0][0], ", ", RGB[80][0], ", ", RGB[0][50])
    print("RGB before shape: ", RGB.shape)
    temp_rgb = np.transpose(RGB, (2, 0, 1))
    print("RGB middle shape: ", temp_rgb.shape)
    temp = temp_rgb[0].copy()
    temp_rgb[0] = temp_rgb[2].copy()
    temp_rgb[2] = temp
    RGB = np.transpose(temp_rgb, (1,2,0))
    print("some vals in RGB after switching r and b: ", RGB[0][0], ", ", RGB[80][0], ", ", RGB[0][50])
    print("rgb 244 before powering: ", RGB[244][17])
    # RGB = RGB**0.4
    print("some vals in RGB after powering by 0.4: ", RGB[0][0], ", ", RGB[80][0], ", ", RGB[0][50])
    print("RGB after shape: ", RGB.shape)
    z = max(RGB[244][17])
    print("z: ", z)
    print("rgb 244: ", RGB[244][17])
    RGB_clip = np.where(RGB>z, z, RGB)
    print("rgbclip after getting rid of outliers: ", RGB_clip[0][0], ", ", RGB_clip[120][110], ", ", RGB_clip[0][50])
    RGB_clip = RGB_clip/z
    print("rgbclip after dividing: ", RGB_clip[0][0], ", ", RGB_clip[120][110], ", ", RGB_clip[0][50])
    RGB = RGB_clip**0.4
    RGB_int =  cv.normalize(RGB, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    print("rgbclip after powering: ", RGB[0][0], ", ", RGB[120][110], ", ", RGB[0][50])
    print("RGB shape: ", RGB.shape)
    print("max of rgb: ", np.max(RGB))
    print("min of rgb: ", np.min(RGB))
    cv.imshow("Out", RGB)
    cv.imshow("int", RGB_int)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def main():
    ref4_dict = loadmat("./assets/ref4_scene4.mat")
    reflectances = ref4_dict['reflectances']
    # Possible Procedure: Display a hyperspectral image
    # Possible Procedure: Plot a graph of the reflectance spectrum at a pixel

    illum_dict = loadmat('./assets/illum_25000.mat')
    illum = illum_dict['illum_25000']

    xyzbar_dict = loadmat("./assets/xyzbar.mat")
    xyzbar = xyzbar_dict['xyzbar']
    illuminate(reflectances, illum, xyzbar)

if __name__ == "__main__":
    main()