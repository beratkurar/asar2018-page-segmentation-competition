import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import os


IMAGE_REGION_LABEL = 'ImageRegion'
GRAPHIC_REGION_LABEL = 'GraphicRegion'
TEXT_REGION_LABEL = 'TextRegion'

IMAGE_REGION_VALUE = 128
TEXT_REGION_VALUE = 0
EMPTY_REGION_VALUE = 255

images_path = 'train/'

#images_path='ygraphics/'
flag=0
for path in os.listdir(images_path):

        if path[-3:]=='jpg':
    
            xml_path = path[:-3]+'xml'
        
            img = cv2.imread(images_path+path)
            tree = ET.parse(images_path+xml_path)
            root = tree.getroot()
        
            page = root[1]
        
            number_of_regions = len(page)
            rec_text_regions = []
            rec_image_regions = []
            poly_text_regions = []
            poly_image_regions = []
            for i in range(1, number_of_regions):
                tag = page[i].tag
        
                if TEXT_REGION_LABEL not in tag and IMAGE_REGION_LABEL not in tag and GRAPHIC_REGION_LABEL not in tag:
                    continue
        
                points = page[i][0].attrib.get('points').split(' ')
                number_of_vertices = len(points)
                vertices_list = []
                for j in range(number_of_vertices):
                    x = int(points[j].split(',')[0])
                    y = int(points[j].split(',')[1])
                    vertices_list.append([x, y])
                if TEXT_REGION_LABEL in tag:
                    if len(vertices_list) == 4:
                        rec_text_regions.append(vertices_list)
                    elif len(vertices_list) > 4:
                        poly_text_regions.append(vertices_list)
                elif IMAGE_REGION_LABEL in tag or GRAPHIC_REGION_LABEL in tag:
                    if len(vertices_list) == 4:
                        rec_image_regions.append(vertices_list)
                    elif len(vertices_list) > 4:
                        poly_image_regions.append(vertices_list)
        
        
        
            labeled = np.ones((img.shape[0], img.shape[1])) * EMPTY_REGION_VALUE
        
            text_pts = np.array(rec_text_regions, dtype=np.int32)
            images_pts = np.array(rec_image_regions, dtype=np.int32)
        
            labeled = cv2.fillPoly(labeled, images_pts, IMAGE_REGION_VALUE)
            labeled = cv2.fillPoly(labeled, text_pts, TEXT_REGION_VALUE)
        
            for poly in poly_image_regions:
                labeled = cv2.fillPoly(labeled, np.array([poly], dtype=np.int32), IMAGE_REGION_VALUE)
        
            for poly in poly_text_regions:
                labeled = cv2.fillPoly(labeled, np.array([poly], dtype=np.int32), TEXT_REGION_VALUE)
        
        
            #cv2.imwrite('textandimages/'+path, img)
            cv2.imwrite('ltrain/' + path, labeled)
    
