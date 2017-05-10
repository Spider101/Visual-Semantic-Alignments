import json
import numpy as np
data = json.load(open('combined_data.json'))
import cv2
import pdb
import os

"""
Change the 'root_path' to whichever folder comtains all your images

for.eg, if your images are inside a folder called 'pics'
then root_path = 'pics/'

"""
root_path = 'images/'

image2regions = {}
region2phrase = {}
region2crop = {}


"""
Initializing and building storage structure
Will take time to run

"""

imageIds = []

for d in data:
    image_id = d['id']
    imageIds.append(image_id)
    if type(image_id) == int:
        image_id = str(image_id)

    image2regions[image_id] = []
    for r in d['regions']:
        region_id = r['region_id']
        image2regions[image_id].append(region_id)
        region2phrase[region_id] = r['phrase']
        region2crop[region_id] = {'x':r['x'], 'y':r['y'], 'h':r['height'], 'w':r['width']}
    
def read_image(image_id):

    """ Given an image id as a string or int, return the image"""
    if type(image_id) == int:
        image_id = str(image_id)

    image = cv2.imread(root_path+image_id+'.jpg')
    return image
    

def get_crops(image_id, visualize=False):

    """
    image_id: str or int
    visualize: Boolean, to visualize the crops. Useful when testing


    returns a zipped list of crops and corresponding phrases.
    each entry in the returned list is a tuple (crop: numpy array, phrase: text)

    """
    if type(image_id) == int:
        image_id = str(image_id)
   
    image = read_image(image_id)

    imgPath = os.path.join(root_path, image_id)
    if not os.path.exists(imgPath): os.makedirs(imgPath)

    crops, phrases = [], []

    for region in image2regions[image_id]:
        y,x,w,h = region2crop[region].values()
        crop = image[y:y+w,x:x+h]
        cv2.imwrite(os.path.join(imgPath,'crop_%s.jpg' % region), crop)
	print imgPath,region

        crops.append(crop)

    crops = np.array(crops)
    return crops


for image_id in map(str,imageIds):
    crops = get_crops(image_id, visualize=False)
