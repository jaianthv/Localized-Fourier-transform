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


def prepare_data(image, Padd):
    ### to convert the image image into 1D array with neighbouring pixels
    image_shape = np.shape(image)
    #y-axis - 0
    #x-axis - 1
    
    ### step one - resize
    xx = np.ravel(image)
    #print (xx)
    
    ### pad 4 zeros on either side
    xx = np.pad(xx, (Padd,Padd), 'constant', constant_values=(0,0))

    ### split multiple arrays with 4 + pixel + 4
    ## Here we create a list where each indices has the pixel of interest in the center.
    
    Rearranged_matrix = []
    for i in range(-1*(Padd), Padd+1):
        Rearranged_matrix.append(shift(xx,i))
    
    New_array = Rearranged_matrix[0]
    for i in range(1, len(Rearranged_matrix)):
        New_array = np.vstack((New_array, Rearranged_matrix[i]))
    
   
    return New_array





def get_image_FFT(FFT_data,Padd,img_ref): # to test image normalization by FFT
    
    IFFT = np.fft.ifft(FFT_data, axis=1)
    
            
    IFFT_data = IFFT[:,Padd].real

    result_2 = np.delete(IFFT_data,np.s_[0:Padd])
    result_2 = np.delete(result_2,np.s_[-1-Padd:-1])
    result_2 = result_2.reshape(np.shape(img_ref))
    #plt.matshow(result_2)
    #plt.show()
    return result_2


def get_image_NFFT(image_1D, Padd, img_ref):
    
    result = np.delete(image_1D,np.s_[0:Padd])
    result_1D = np.delete(result,np.s_[-1-Padd:-1])
    result_2D = result_1D.reshape(np.shape(img_ref))
    return result_2D

def get_image_AFFT(FFT_data,Padd,img_ref):
    #IFFT = np.fft.ifft(FFT_data, axis=1)
    IFFT = FFT_data
    IFFT_data = IFFT
    print (len(IFFT_data))
    IFFT_data = IFFT[:,1].real

    result_2 = np.delete(IFFT_data,np.s_[0:Padd])
    result_2 = np.delete(result_2,np.s_[-1-Padd:-1])
    result_2 = result_2.reshape(np.shape(img_ref))
    #plt.matshow(result_2)
    #plt.show()
    return result_2




def get_filter(Padd):
    x = np.array([1,0,1])
    return np.pad(x, (Padd-1,Padd-1),'constant')



def get_only_3(FFT_data, Padd):
    return FFT_data[:,[Padd-1,Padd,-1*Padd]]
    


def get_filter_delete_center(Padd):
    x = np.array([0,0,0])
    return np.pad(x, (Padd-1,Padd-1),'constant', constant_values=(1,1))

def get_frequency(FFT_data_1, Padd):
    Freq = np.fft.fftfreq(len(FFT_data_1[0]))
    Freq = 2*np.pi * np.fft.fftshift(Freq)*-1
    print (Freq)
    Nx = len(FFT_data_1[0])
      
    # calculate frequencies
    dqx = (2 * np.pi)/ ((Nx))
    xF = (np.arange(0, Nx) - math.floor(Nx / 2)) *dqx
    filtx = (1j * xF)/(xF**2)
    print (xF)
    return Freq #xF*1j*-1


def get_harmonic(FFT_data, Padd, position):
    if position > Padd:
        position =1
        print ("Not possible just taking the first harmonic")

    # [-2,-1,0,1,2]
    # len = (Padd*2) +1
    # options on left [0,Padd-1]
    # options on right [-1, -Padd]
    # [0,1,2, 3, 4,5,6]
    # [0,1,2,3,4,  5,  6,7,8,9,10]
    # [0,1,2,3,4,5,6, 7, 8,9,10,11,12,13,14,15]
    Frequency_negative = (Padd) - position
    Frequency_positive = Padd + position
    #print (Frequency_positive)
    #print (Frequency_negative)
    L = FFT_data[:,Frequency_negative]
    R = FFT_data[:,Frequency_positive]
    M = FFT_data[:,Frequency_negative]*0
    XX = np.column_stack((L,M))
    #print (len(FFT_data))
    #print (len(L))
    YY = np.column_stack((XX,R))
    #print (YY[5000])
    return YY
    
def get_harmonic_filter(Padd, position):
    Frequency_negative = (Padd) - position
    Frequency_positive = Padd + position
    Filter = np.zeros((2*Padd)+1)
    
    Filter[Frequency_negative: Frequency_positive+1] =1
    Filter[Padd] =0
    print (Filter)
    return Filter


def distor_image_horizontal(image1, image2, Padd):
    data_1 = prepare_data(image1, Padd)
    data_1 = np.fliplr(data_1.T)
    data_2 = prepare_data(image2, Padd)
    data_2 = np.fliplr(data_2.T)
    
    
        
    FFT_data_1 = np.fft.fftshift(np.fft.fft((data_1), axis=1))
    FFT_data_2 = np.fft.fftshift(np.fft.fft((data_2), axis=1))
        
    slope = np.fft.fftshift(np.fft.fft(np.diff((data_1),axis=1, append=1)))
    print (slope[5000])
    Img_FFT_diff = np.fft.fftshift(np.fft.fft((data_2-data_1), axis=1))
    print (Img_FFT_diff[5000])
    
    
    Img_FFT_diff=  Img_FFT_diff/slope 
    #print (Img_FFT_diff[5000])

    
    
    Img_FFT_diff = get_harmonic(Img_FFT_diff, Padd, 1)
    #Img_FFT_diff = Img_FFT_diff* [1,1,1,1,1,0,1,1,1,1,1]
    
    II = np.fft.ifft(np.fft.ifftshift(Img_FFT_diff))
    
    
    II = get_image_AFFT(II, Padd, image1)
    II = (II.real)
    
    #cv.imwrite("test_a.tiff",II)
    #plt.matshow(II)#, norm=LogNorm()) #/image_1) #, norm=LogNorm())
    #plt.show()
    
    return II




def distor_image_vertical(image1, image2, Padd):
    
    image1 = np.rot90(image1, axes=(1,0))
    image2 = np.rot90(image2, axes=(1,0))

    data_1 = prepare_data(image1, Padd)
    data_1 = np.fliplr(data_1.T)
    data_2 = prepare_data(image2, Padd)
    data_2 = np.fliplr(data_2.T)

    FFT_data_1 = np.fft.fftshift(np.fft.fft((data_1), axis=1))
    FFT_data_2 = np.fft.fftshift(np.fft.fft((data_2), axis=1))
        
    slope = np.fft.fftshift(np.fft.fft(np.diff((data_1),axis=1, append=1)))
   
    Img_FFT_diff = np.fft.fftshift(np.fft.fft((data_2-data_1), axis=1))


    Img_FFT_diff=  Img_FFT_diff/slope

    Img_FFT_diff = get_harmonic(Img_FFT_diff, Padd, 1)
    
    II = np.fft.ifft(np.fft.ifftshift(Img_FFT_diff))
    
    
    II = get_image_AFFT(II, Padd, image1)
    II = (II.real)
    II =  np.rot90(II, axes=(0,1))
    #cv.imwrite("After_iFFT_10.tiff",II)
    #plt.matshow(II) #, norm=LogNorm()) #/image_1) #, norm=LogNorm())
    #plt.show()
    
    return II






############# Dark Field imaging #############################



def get_harmonic_DF_v2(FFT_data, Padd, position, ref):
    if position > Padd:
        position =1
        print ("Not possible just taking the first harmonic")

    #Ref = np.tile(ref[:,Padd], (len(FFT_data[1]),1)).T
    #print (np.shape(Ref))
    Den = (FFT_data[:,Padd]) #/ref[:,Padd])
    Den1 = np.tile(Den, (len(FFT_data[1]),1)).T
    #FFT_data_1 =  (np.absolute(FFT_data)/Ref)/Den1
    FFT_data_1 =  (np.absolute(FFT_data)/ Den1) #/np.absolute(Ref))/Den1
    #FFT_data_1[:,Padd] = Den 
    
    return FFT_data_1




    



def distor_image_DF(image1, image2, Padd):
    data_1 = prepare_data(image1, Padd)
    data_1 = np.fliplr(data_1.T)
    data_2 = prepare_data(image2, Padd)
    data_2 = np.fliplr(data_2.T)
    
    data_1_std = np.std(data_1, axis=1)
    data_2_std = np.std(data_2, axis=1)

    im_temp = get_image_NFFT(data_2_std/data_1_std, Padd, image1)
    #plt.matshow(im_temp)
    #plt.show()
    
    FFT_data_1 = np.fft.fftshift(np.fft.fft((data_1), axis=1))
    FFT_data_2 = np.fft.fftshift(np.fft.fft((data_2), axis=1))



    Sub_single = get_harmonic_DF_v2(FFT_data_2, Padd, 1,FFT_data_1)
    
     
    II = np.fft.ifft(np.fft.ifftshift(Sub_single)).real 
    

    II = get_image_AFFT(II, Padd, image1)
    II = (II.real)*-1
    
    
    #cv.imwrite("DF_Hor_image_1a_new.tiff", II)
    #cv.imwrite("withoutSTD_image_withoutmean.tiff", II) #/np.median(II))
   
  
    
    return II


def distor_image_DF_vertical(image1, image2, Padd):
    
    image1 = np.rot90(image1, axes=(1,0))
    image2 = np.rot90(image2, axes=(1,0))

    data_1 = prepare_data(image1, Padd)
    data_1 = np.fliplr(data_1.T)
    data_2 = prepare_data(image2, Padd)
    data_2 = np.fliplr(data_2.T)

    FFT_data_1 = np.fft.fftshift(np.fft.fft((data_1), axis=1))
    FFT_data_2 = np.fft.fftshift(np.fft.fft((data_2), axis=1))
        
   
    #Sub_single = FFT_data_2 *(1/FFT_data_1[:,Padd])
    Sub_single = get_harmonic_DF_v2(FFT_data_2, Padd, 1,FFT_data_1)

    
     
    II = np.fft.ifft(np.fft.ifftshift(Sub_single)).real 
    
    
    II = get_image_AFFT(II, Padd, image1)
    II = (II.real)
    II =  np.rot90(II, axes=(0,1))*-1
    #cv.imwrite("DF_vert_image_1a_new.tiff",II)
    
    return II


def get_dark_field(image1, image2, Padd):
    A = distor_image_DF(image1, image2, Padd)
    B = distor_image_DF_vertical(image1, image2, Padd)
    C = A+B/2
    C = (np.absolute((C)-np.median(C)))
    return C
