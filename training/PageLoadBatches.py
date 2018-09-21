# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:20:20 2017

@author: B
"""
import sys
sys.path.append('/root/PageSegComp/Models/')
import numpy as np
import cv2
import glob
import itertools


def getImageArr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):

    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img = img/255.0

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
            return img
    except Exception as e:
        print (path)
        print (e)
        img = np.zeros((  height , width  , 3 ))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img



def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, ( width , height ))
        img = img[:, : , 0]
        img[img<50]=0
        img[img>200]=255
        img[(img>100)&(img<150)]=128
        

		#for c in range(nClasses):
		#	seg_labels[: , : , c ] = (img == c ).astype(int)
        seg_labels[: , : , 0 ] = (img == 255 ).astype(int)
        seg_labels[: , : , 1 ] = (img == 128).astype(int)
        seg_labels[: , : , 2 ] = (img == 0).astype(int)
    except Exception as e:
        print (e)
		
    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels



def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
	
	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
	segmentations.sort()

	assert len( images ) == len(segmentations)
	for im , seg in zip(images,segmentations):
		assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

	zipped = itertools.cycle( zip(images,segmentations) )

	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = next(zipped)
			X.append( getImageArr(im , input_width , input_height )  )
			Y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )

		yield np.array(X) , np.array(Y)

def testSet( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
	
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
    images.sort()
    segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
    segmentations.sort()

    assert len( images ) == len(segmentations)
    for im , seg in zip(images,segmentations):
        assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

    zipped = itertools.cycle( zip(images,segmentations) )
    X = []
    Y = []
    for i in range(0,245):
        im , seg = next(zipped)
        X.append( getImageArr(im , input_width , input_height )  )
        Y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )

    return np.array(X) , np.array(Y)



def test(pageimage,model):
    return 0

# import Models , LoadBatches
# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 
# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
# m.fit_generator( G , 512  , nb_epoch=10 )

