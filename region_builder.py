import json
import numpy as np
import cv2
import pdb
from os.path import join, exists
from os import mkdir
from tqdm import *

from config.resources import local_data_path, METADICT_FNAME

""" Given an image id as a string or int, return the image"""
def read_image(image_id, img_dir):
    return cv2.imread(join(img_dir, image_id+'.jpg'))

'''crop out the regions in the individual images in the dataset'''
def make_crops(image_id, regions, source_dir, dest_dir):

    image = read_image(image_id, source_dir)

    img_path = join(dest_dir, image_id)
    if not exists(img_path): 
        mkdir(img_path)

    crops = []
    for region in tqdm(regions, total=len(regions)):
        
        x, y, w, h, reg_id = region["x"], region["y"], region["width"], \
                                region["height"], str(region["region_id"])
        crop = image[y:y+w, x:x+h]
        crop_path = join(img_path, reg_id + ".jpg")
        cv2.imwrite(crop_path, crop)

