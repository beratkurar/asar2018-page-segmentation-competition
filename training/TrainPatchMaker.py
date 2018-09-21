# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:43:28 2017

@author: B
"""
import random
import cv2
import os

patchSize=320
patchNumber=0
folder='train/'
lfolder='ltrain/'
i=0
while (i <100000):
    pages=os.listdir(folder)
    page_number=random.randint(0,590)
    page_name=pages[page_number]
    page=cv2.imread(folder+page_name,3)
    lpage=cv2.imread(lfolder+page_name[:-3]+'jpg',0)
    rows,cols,ch=page.shape
    x=random.randint(0,rows-patchSize)
    y=random.randint(0,cols-patchSize)
    lpatch=lpage[x:x+patchSize,y:y+patchSize]
    cv2.imwrite("p"+lfolder+page_name[:-4]+"_patch"+str(i)+".png",lpatch)
    patch=page[x:x+patchSize,y:y+patchSize]
    cv2.imwrite("p"+folder+page_name[:-4]+"_patch"+str(i)+".png",patch)
    print(i)
    i=i+1
