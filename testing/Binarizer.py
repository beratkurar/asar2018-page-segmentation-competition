# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 07:37:18 2018

@author: B
"""

import cv2
import os

infolder='testb/'
outfolder='btestb/'

for f in os.listdir(infolder):
    img=cv2.imread(infolder+f,0)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite(outfolder+f,th2)