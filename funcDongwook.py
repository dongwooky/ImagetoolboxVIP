##from PyQt5.QtWidgets import *
##from PyQt5 import uic
##from PyQt5.QtGui import *
import cv2
import numpy as np

fscale=5

#grayscale
def Grayscale(image_arr):
   
    arr=[0.299, 0.587, 0.114]
    gray_arr=image_arr.dot(arr)
    
    return gray_arr

#gaussian
def Gaussian(gray_arr):

    h_scale=int(fscale/2)

    filter_arr=[[1,4,7,4,1],
                [4,16,26,16,4],
                [7,26,41,26,7],
                [4,16,26,16,4],
                [1,4,7,4,1]]
    filter_arr/273

    pad_scale=int(fscale/2)
    pad=((pad_scale,pad_scale),(pad_scale,pad_scale))
    image_pad = np.pad(gray_arr, pad, constant_values=(128))

    x,y=gray_arr.shape

    gaussian_arr=np.zeros((x,y))

    for i in range(pad_scale,y+pad_scale):
        for j in range(pad_scale,x+pad_scale):
            sample_pad=image_pad[j:j+fscale,i:i+fscale]
            gaussian=sample_pad*filter_arr
            gaussian_arr[j-pad_scale,i-pad_scale]=gaussian.sum()
    
    return gaussian_arr

#laplacian_filter
def laplacian_filter(gray_arr):

    laplacian_arr4=[[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]]
    laplacian_arr8=[[-1,-1,-1],
                    [-1,8,-1],
                    [-1,-1,-1]]
        
    pad=((1, 1),(1, 1))
    image_pad=np.pad(gray_arr, pad, constant_values=(128))

    x, y=gray_arr.shape

    laplacian_arr=np.zeros((x,y))

    for i in range(1, 1+y):
        for j in range(1,1+x):
            sample_pad=image_pad[j:j+3,i:i+3]
            laplacian=np.dot(sample_pad,laplacian_arr4)
            laplacian_arr[j-1, i-1]=laplacian.sum()

    for i in range(0,y):
        for i in range(0,x):
            if(laplacian_arr[j,i]*laplacian_arr[j+1,i]<0):
                laplacian_arr[j,i]=255

    return laplacian_arr

if __name__=="__main__":
    src=cv2.imread("data/fruits.png")
    image=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)

    gray_image=Grayscale(image)
    gaussian_image=Gaussian(image)

    cv2.imshow('gray_image',gray_image.astype(np.uint8))
    cv2.imshow('gaussian_image',gaussian_image.astype(np.uint8))
    cv2.waitKey(0)
    # cv2.destroyAllWindows()