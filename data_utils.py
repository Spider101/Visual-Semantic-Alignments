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

from text_utils import build_vocab
from model_utils import encode_text
from config.resources import local_data_path, METADICT_FNAME
from utils import load_pkl, dump_pkl

'''build the dataset for the experiments from the metadata dictionary'''
def build_text_dataset(metadata):

    dataset = {}

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
    
    #TODO: figure out what to do here
    return dataset             

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--op", help="operation to perform")
    parser.add_argument("--modality", default="text", help="the modality to work on")
    parser.add_argument("--vocab_lim", default=None, help="size of vocabulary to use")

    args = parser.parse_args()
    
    with open(join(local_data_path, METADICT_FNAME)) as f:
        meta_dict = json.load(f)

    if args.op == "build_dataset":
        
        if args.modality == "text":
            
            text_dataset = build_text_dataset(meta_dict)
            dump_pkl(text_dataset, local_data_path, "text_feats")
    
    elif args.op == "build_vocab":

        vocab = build_vocab(meta_dict, args.vocab_lim)
        fname = "vocab_full" if args.vocab_lim is None else "vocab_" + args.vocab_lim
        dump_pkl(vocab, local_data_path, fname)
