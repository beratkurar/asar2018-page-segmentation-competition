# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 07:37:18 2018

@author: B
"""
import cv2
import os
import numpy as np

def det(img):


    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(img)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)


    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)

    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols / 20

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)


    # [smooth]
    # Inverse vertical image
    #horizontal = cv.bitwise_not(horizontal)
    res=img+horizontal   

    return res

infolder='tbtestb/'
outfolder='nltbtestb/'
patchsize=320

for f in os.listdir(infolder):
    img=cv2.imread(infolder+f,0)
    nimg=det(img) 
    cv2.imwrite(outfolder+f,nimg)




