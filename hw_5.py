# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:39:15 2021

@author: USER
"""
# Import neccessary library
import cv2
import numpy as np
from matplotlib import pyplot as plt

#read image
img = cv2.imread('img/bird.png',0)  
plt.imshow(img,cmap = 'gray')

#Fourier function 
f = np.fft.fft2(img)
#define magnitude Spectrum for visualization
magnitude_spectrum = 20*np.log(np.abs(f))
magnitude_spectrum = np.array(magnitude_spectrum, dtype=np.uint8)
plt.imshow(magnitude_spectrum, cmap = 'gray')

#Fourier Shift transform
fshift = np.fft.fftshift(f)
msshift = 20*np.log(np.abs(fshift))
msshift = np.array(msshift, dtype=np.uint8)
plt.imshow(msshift, cmap = 'gray')

#Inverser
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)
# plt.imshow(img_back, cmap='gray')


    
    
