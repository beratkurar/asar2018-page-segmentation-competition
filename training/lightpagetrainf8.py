# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:14:50 2017

@author: B
"""
import sys
sys.path.append('/root/asar2018psc/training/Models/')
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

parser.add_argument("--test_images", type = str , default = "ptest/")
parser.add_argument("--test_annotations", type = str , default = "pltest/")
parser.add_argument("--output_path", type = str , default = "pprediction/")



parser.add_argument("--epochs", type = int, default = 250 )
parser.add_argument("--batch_size", type = int, default = 16 )
parser.add_argument("--val_batch_size", type = int, default = 16 )
parser.add_argument("--test_batch_size", type = int, default = 16 )

parser.add_argument("--load_weights", type = str , default = 'bestweights')

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

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
sgd = optimizers.SGD(lr=0.0001)
#adm=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

m.compile(loss='categorical_crossentropy',
      optimizer= sgd,
      metrics=['accuracy'])


if len( load_weights ) > 0:
    print("loading initial weights")
    m.load_weights(load_weights)


print ( m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G  = PageLoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
	G2  = PageLoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

mcp=ModelCheckpoint( filepath=save_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True,verbose=1)

if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 1  , epochs=1 )
else:
    for ep in range( epochs ):
        m.fit_generator( G , 6250 , validation_data=G2 , validation_steps=930,  epochs=1,callbacks=[mcp] )
