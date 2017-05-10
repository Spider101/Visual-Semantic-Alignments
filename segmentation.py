#Author : Rajveer Nehra
#Description : Crop the image segments
import os
import json
import cv2
from os import mkdir, system, listdir
from os.path import join, exists
from pprint import pprint
import cv2
import errno


root_path = '/Users/rajveernehra/Desktop/Visual-Semantic-Alignments/Data_folder/images/'

def load_Image_Folder(folder):
    images_array = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images_array.append(img)
    return images_array

def read_image(image_id):
    
    """ Given an image id as a string or int, return the image"""
    if type(image_id) == int:
        image_id = str(image_id)
    
    image = cv2.imread(root_path+image_id+'.jpg')
    return image


if __name__ == "__main__":
    image_idx = []
    region_idx = []
    image_iD_region = {}
    crop_regions = {}
    B = []
    cropped_image = []
    Folder = []
    Image_files = []
    image2regions = {}
    phrases = []
    
  
    with open('combined_data.json') as data_file:
        data = json.load(data_file)
        for line in data:
            image_ID = line['id']
            image_idx.append(image_ID)
            region2phrase = {}
            image2regions[image_ID] = []
            for r in line['regions']:
                region_id = r['region_id']
                region_idx.append(region_id)
                phrase_reg = r['phrase']
                phrases.append(phrase_reg)
                region2phrase[region_id] = phrase_reg
                image2regions[image_ID].append(region_id)
                crop_regions[region_id] = {'x':r['x'], 'y':r['y'], 'h':r['height'], 'w':r['width']}
                image_iD_region['imageid'] = image_ID
                image_iD_region['regionid'] = region_id
                A = image_iD_region.values()
                B.append(A)

    for id in image_idx:
        destPath = join('crops',str(id))
        if not exists(destPath):
            mkdir(destPath)
        region_idx = image2regions[id]
        print("The regions are for images : ", id, "and the region", region_id)
        img = read_image(id)
        for region_id in region_idx:
            print 'r:',region_id
            y,x,h,w = crop_regions[region_id].values()
            print("the crop regions : ", x,y,h,w,crop_regions[region_id])
            crop = img[y:y+h,x:x+w]
            imgPath = join(destPath, str(region_id))
            # print imgPath
            cv2.imwrite(imgPath+'.jpg', crop)
            #print region2phrase[region_id]
            #cv2.imshow('region_%s' % region_id, crop)
#cv2.waitKey(0)
       
# print("The number of images are : ", image_idx[i], len(Image_files))
#   for i in range(len(image_idx)):
#       for j in range(len(region_idx)):
#           x,y,h,w = crop_regions[region_idx[j]].values()
#print("the crop regions are : ",y,x,w,h)
#cropped_image_1 = Image_files[i][x:x+w, y:y+h]
#cropped_image.append(cropped_image_1)
#for entry in cropped_image:

            #cv2.imwrite("", cropped_image[j])
#cropped_image[j].save("image_idx[i].jpg")
#print("the size of cropped_regions", len(cropped_image))
#   print("the number of folders ; ", len(Folder))


#Image_files = load_Image_Folder("images_01")
#   print(Image_files[1].shape)
#   cropped_image = []

#   for i in range(len(load_Image_Folder("images_01"))):
#       x,y,h,w = crop_regions[region_idx[i]].values()
#       print(image_idx[i])
        #image_instant = read_image_id(image_idx[i])
#       print("the crop regions are : ",y,x,w,h)
#       cropped_image_1 = Image_files[i][x:x+w, y:y+h]
#       cropped_image.append(cropped_image_1)
#   cv2.imwrite("Image[i].png", cropped_image[1])
#
#cv2.imwrite("cropped.png", image_instant)
  # cv2.imshow("original", Image_files[1])
   #cv2.waitKey(0) ##This is to see the images 
#print(data[1]) 
