# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:43:28 2017

@author: B
"""

import cv2
import os
patchSize=320
patchNumber=0
folder='nltbtestb/'
for filename in os.listdir(folder):

    page=cv2.imread(folder+filename,1)
    rows,cols,ch=page.shape
    #col_remainder=cols%patchSize
    i=0
    for x in range(0,rows,patchSize):
        for y in range(0,cols,patchSize):
            
            
            patch=page[x:x+patchSize,y:y+patchSize]
            cv2.imwrite('ptestb/'+filename[:-4]+"_patch"+str(patchNumber)+".png",patch)
            patchNumber=patchNumber+1
            
            
