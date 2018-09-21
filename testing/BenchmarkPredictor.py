# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:14:50 2017

@author: B
"""
import sys
sys.path.append('/root/PageSegComp/Models/')
import numpy as np
np.random.seed(123)
import argparse
import Models , PageLoadBatches
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import glob
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str,default ="bestweights"  )
parser.add_argument("--train_images", type = str, default ="ptrain/"  )
parser.add_argument("--train_annotations", type = str, default = "pltrain/"  )
parser.add_argument("--n_classes", type=int, default = 3 )
parser.add_argument("--input_height", type=int , default = 320  )
parser.add_argument("--input_width", type=int , default = 320 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "pvalidation/")
parser.add_argument("--val_annotations", type = str , default = "plvalidation/")

parser.add_argument("--test_images", type = str , default = "pbench/")
parser.add_argument("--test_annotations", type = str , default = "pltest/")
parser.add_argument("--output_path", type = str , default = "pprediction/")



parser.add_argument("--epochs", type = int, default = 250 )
parser.add_argument("--batch_size", type = int, default = 16 )
parser.add_argument("--val_batch_size", type = int, default = 16 )
parser.add_argument("--test_batch_size", type = int, default = 16 )

parser.add_argument("--load_weights", type = str , default = '')

parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--optimizer_name", type = str , default = "sgd")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
test_images_path = args.test_images
test_segs_path = args.test_annotations
test_batch_size = args.test_batch_size


model_name = args.model_name


modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )


output_height = m.outputHeight
output_width = m.outputWidth

print('loading test images')
images = sorted(glob.glob( test_images_path + "/*.jpg"  ) + glob.glob( test_images_path+ "/*.png"  ) +  glob.glob( test_images_path + "/*.jpeg"  ))
images.sort()
print('loading the last best model')
loaded_model=modelFN( n_classes , input_height=input_height, input_width=input_width)
loaded_model.load_weights('bestweights02167')
colors=[(255,255,255),(0,0,255),(255,0,0)]

for imgName in images:
    outName = imgName.replace( test_images_path ,  args.output_path)
    X = PageLoadBatches.getImageArr(imgName , args.input_width  , args.input_height)
    pr = loaded_model.predict( np.array([X]))[0]
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
    seg_img = np.zeros( ( output_height , output_width , 3  ) )
    for c in range(n_classes):
        seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    seg_img = cv2.resize(seg_img , (input_width , input_height ))
    cv2.imwrite(outName , seg_img )

print('combining the predictions')
patchSize=320
patchNumber=0
predictions='pprediction/'
original='lbench/'

paths = sorted(glob.glob(predictions+ "*.png" ))
pages=[item.split('_patch')[0] for item in paths]
oldpage=pages[0]
g=[[]]
i=0
pathc=0
for page in pages:
    if page==oldpage:
        g[i].append(paths[pathc])
        pathc=pathc+1
    else:
        i=i+1
        g.append([])
        g[i].append(paths[pathc])
        pathc=pathc+1
        oldpage=page

     
for group in g:
    group=np.array(group)
    ord_indices=[]
    for i in  range(0,len(group)):
        ord_indices.append(int(group[i].split('_patch')[1].split('.')[0]))
    order = np.argsort(ord_indices)
    group = group[order]
    oi=group[0].split('/')[1].split('_patch')[0]+'.jpg'
    originalPage=cv2.imread(original+oi,0)
    rows,cols=originalPage.shape
    x=rows//patchSize
    y=cols//patchSize
    sx=x*patchSize
    sy=y*patchSize
    ni=np.zeros((int(sx),int(sy),3))+255
    cp=0
    for i in range(0,sx,patchSize):
        for j in range(0,sy,patchSize):
            ni[i:i+patchSize,j:j+patchSize]=cv2.imread(group[cp],1)
            cp=cp+1
    cv2.imwrite('out/'+group[0].split('/')[1].split('_patch')[0]+'.png',ni)

