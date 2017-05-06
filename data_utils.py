###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Semantic Alignments
# Date Created: 4/12/2017
# 
# File Description: This script contains helper methods for building, curating
# and manipulating the dataset
###############################################################################

from __future__ import print_function
import json
from os.path import join, exists
from tqdm import *
import argparse
import pdb

from skip_thoughts import skipthoughts
from model_utils import encode_text
from config.resources import data_path, METADICT_FNAME
from utils import load_pkl, dump_pkl

'''define sentence encoder and load it with pretrained weights'''
def build_sent_encoder():

    model = skipthoughts.load_model()
    return skipthoughts.Encoder(model)

'''build the dataset for the experiments from the metadata dictionary'''
def build_text_dataset(metadata):

    dataset = {}

    #set up the encoder to be used to compute the textual embeddings for
    #the captions 
    text_encoder = build_sent_encoder()

    #iterate over the items in the metadata
    for idx in trange(len(metadata)):
        
        #collect relevant metadata from the current item
        regions = metadata[idx]["regions"]
        img_id = str(metadata[idx]["id"])
        paragraph = metadata[idx]["paragraph"]
        
        if img_id not in dataset.keys():
            dataset[img_id] = {}

        captions = []
        #iterate over the regions corresponding to each item (image)
        for reg_idx in trange(len(regions)):

            #make sure the regions belong to the current image
            if int(img_id) != regions[reg_idx]["image_id"]:
                pdb.set_trace()

            caption = regions[reg_idx]["phrase"]
            
            #safety check to prevent blank strings being passed to the encoder
            if caption.replace(" ", "") == "":
                caption = "\b"

            captions.append(caption)

        sent_feats = encode_text(text_encoder, captions)

        for reg_idx in trange(len(regions)):

            region_id = str(regions[reg_idx]["region_id"])

            if region_id not in dataset[img_id].keys():
                dataset[img_id][region_id] = {}

            dataset[img_id][region_id] = {"phrase": regions[reg_idx]["phrase"], 
                                            "feature": sent_feats[reg_idx, :]}

    return dataset             

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--op", help="operation to perform")
    parser.add_argument("--modality", default="text", help="the modality to work on")

    args = parser.parse_args()

    if args.op == "build_dataset":
        
        if args.modality == "text":
            
            with open(join(data_path, METADICT_FNAME)) as f:
                meta_dict = json.load(f)
            
            text_dataset = build_text_dataset(meta_dict)
            
            dump_pkl(text_dataset, data_path, "text_feats")
