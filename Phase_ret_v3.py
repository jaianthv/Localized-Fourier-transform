import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.fft import fft, ifft, fftfreq
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import shift
from matplotlib.colors import LogNorm
from skimage import feature
import math
from scipy import signal
from scipy.ndimage.filters import gaussian_filter




def get_frequency(FFT_data_1):
    Freq = np.fft.fftfreq(len(FFT_data_1[0]))
    Freq = 2*np.pi * np.fft.fftshift(Freq)*-1
    Freq = np.where(Freq!=0, Freq,1)
    print (Freq)
    Nx = len(FFT_data_1[0])
      
    # calculate frequencies
    dqx = (2 * np.pi)/ ((Nx))
    xF = (np.arange(0, Nx) - math.floor(Nx / 2)) *dqx
    filtx = (1j * xF)/(xF**2)
    print (xF)
    
    return Freq #xF*1j*-1


def get_phase1(grad_x, grad_y):


    
    
    
    

    image1 = np.rot90(grad_y, axes=(1,0))
   
    for_0_x =np.fft.fft(grad_x)
    for_0_y = np.fft.fft(image1)
    slope_x = np.fft.fftshift(np.fft.fft(((grad_x))))
    slope_y = np.fft.fftshift(np.fft.fft(((image1))))
    
    Freq1 = get_frequency(slope_x)*1j
    Freq2 = get_frequency(slope_y)*1j
   
    Freq_x = np.tile(Freq1, (len(slope_x[:,1]),1))
    Freq_y = np.tile(Freq2, (len(slope_y[:,1]),1))
    
    XS = np.shape(Freq_x)
    YS = np.shape(Freq_y)
    
    X_0 = for_0_x[:,0]
    Y_0 = for_0_y[:,0]
       
    slope_x1 = (slope_x/Freq_x) 
    slope_y1 = (slope_y/Freq_y) 

    slope_x1 = np.fft.ifftshift(slope_x1)
    slope_y1 = np.fft.ifftshift(slope_y1)
    
    for i in range(len(X_0)):
        slope_x1[i,0] = slope_x1[i,0] + np.pi*X_0[i]
    for i in range(len(Y_0)):
        slope_y1[i,0] = slope_y1[i,0] + np.pi*Y_0[i] 
    
    slope_ix = np.fft.ifft(slope_x1)
    slope_iy = np.fft.ifft(slope_y1)
   

    II_x = slope_ix
    II_y = slope_iy
    II_y = np.rot90(II_y, axes=(0,1))
    
   
    II = ((II_x.real + II_y.real)/2)
    
    return II



