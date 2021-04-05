# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:24:32 2021

@author: USER
"""

import cv2
import numpy as np
from math import sqrt,exp
from matplotlib import pyplot as plt

img = cv2.imread('img/bird.png',0)
plt.imshow(img,cmap = 'gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift=np.fft.fftshift(dft)
magnitude_spectrum= 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.imshow(magnitude_spectrum, cmap = 'gray')


#create kernel
print(img.shape)
kernel_size=50
base=np.zeros(img.shape[:2])
rows, cols = img.shape[:2]
crow, ccol = (rows//2, cols//2)
#create mask first, center square ->1, the whole remain zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-kernel_size:crow+kernel_size, ccol-kernel_size:ccol+kernel_size] = 1  

def apply_filter(mask, dft_shiftshift):
    #magnitude
    magnitude_spectrum= 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    #apply mask and inverse dft
    fshift= dft_shift*mask
    
    #apply to magnitude
    magnitude_mask= magnitude_spectrum * mask[...,0]
    
    #ifft transf
    
    f_ishift=np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    
    #magnitude
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    
    #plot result
    plt.figure(figsize=(6.0*5, 37.5*5), constrained_layout=False)
    plt.subplot(1, 3, 1)
    plt.imshow(mask[...,0], cmap='gray'), plt.title("mask")
    plt.subplot(1,3, 2)
    plt.imshow(magnitude_mask, cmap='gray'), plt.title("low/high pass filter")
    plt.subplot(1, 3, 3)
    plt.imshow(img_back, cmap='gray'), plt.title("filtered image")
    
apply_filter(mask, dft_shift)

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

#idea low pass


def ideaLPFilter(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base
    
    
mask = ideaLPFilter(50, img.shape)
mask = np.dstack([mask, mask])
apply_filter(mask, dft_shift)

#Butterworth low pass
def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base
mask = butterworthLP(50, img.shape, 3)
mask = np.dstack([mask, mask])
apply_filter(mask, dft_shift)


#Guassian low pass
def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base
mask = gaussianLP(50, img.shape)
mask = np.dstack([mask, mask])
apply_filter(mask, dft_shift)



#high past
mask = 1 - mask
apply_filter(mask, dft_shift)


#Idealhighpass
def ideaHPFilter(D0, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols=imgShape[:2]
    center=(rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((x,y), center) < D0:
                base[y,x]= 0
    return base

mask = ideaHPFilter(50, img.shape)

mask = np.dstack([mask, mask])
apply_filter(mask, dft_shift)

#butter highpass
def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

mask = butterworthHP(50, img.shape, 3)
mask = np.dstack([mask, mask])
apply_filter(mask, dft_shift)

#guassian HP
def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base
mask = gaussianHP(50, img.shape)
mask = np.dstack([mask, mask])
apply_filter(mask, dft_shift)