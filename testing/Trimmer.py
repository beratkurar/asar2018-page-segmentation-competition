# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 07:37:18 2018

@author: B
"""

import cv2
import os
import numpy as np

infolder='btestb/'
outfolder='tbtestb/'
patchsize=320

for f in os.listdir(infolder):
    img=cv2.imread(infolder+f,0)
    rows,cols=img.shape
    wimg=np.ones((rows,cols))*255
    row_crop=(rows/100)*3
    col_crop=(cols/100)*10
    cimg=img[row_crop:rows-col_crop,col_crop:cols-col_crop]
    wimg[row_crop:rows-col_crop,col_crop:cols-col_crop]=cimg
    rows_complete=patchsize-(rows%patchsize)
    cols_complete=patchsize-(cols%patchsize)
    nimg=np.ones((rows+rows_complete,cols+cols_complete))*255
    nimg[0:rows,0:cols]=wimg   

    cv2.imwrite(outfolder+f,nimg)