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
from os.path import join, exists, normpath, basename
from os import listdir, mkdir
from nltk.tokenize import sent_tokenize
from tqdm import *
import argparse
import numpy as np
from numpy.random import randint, choice
import pdb

from text_utils import build_vocab, text_to_seq
from region_builder import read_image, make_crops
from model_utils import build_feat_extractor, extract_feats
from config.resources import (local_data_path, vocab_dict, RANDOM_SEED, 
                                METADICT_FNAME)
from utils import load_pkl, dump_pkl

'''generates batches of data in the textual modality'''
def text_data_gen(data, batch_size=32):
   
    seed(RANDOM_SEED)
    captions = data["captions"]
    
    nb_captions = len(captions)
    max_seq_len = max([len(sent) for sent in captions])
    batch_feats = np.zeros((batch_size, max_seq_len), dtype=int)

    while True:

        for batch_idx in range(batch_size):
            
            curr_idx = randint(0, nb_captions)
            curr_len = len(captions[curr_idx])
            batch_feats[batch_idx, :curr_len] = captions[curr_idx]
        
        yield batch_feats, batch_labels

'''generates batches of data for experiments'''
def data_gen(data, nb_reg=10, nb_feats=4096, batch_size=32):
    
    seed(RANDOM_SEED)
    captions = data["captions"]
    vis_encoder = build_feat_extractor("vis")
    
    nb_captions = len(captions)
    max_seq_len = max([len(sent) for sent in captions])
    crop_dir = join(local_data_path, "crops")
    img_dirs = [ join(crop_dir, fname) for fname in listdir(crop_dir) ]

    batch_vis = np.zeros((batch_size, nb_reg, nb_feats))
    batch_text = np.zeros((batch_size, max_seq_len), dtype=int)

    while True:

        for batch_idx in range(0, batch_size, 2):
            
            #pos samples
            curr_idx = randint(0, nb_captions)
            curr_len = len(captions[curr_idx])
            batch_text[batch_idx, :curr_len] = captions[curr_idx]

            img_dir = join(crop_dir, str(curr_idx))
            reg_paths = [ join(img_dir, fname) for fname in listdir(img_dir)[:nb_reg] ]

            feat_list = []
            for reg_path in reg_paths:
                feat_list.append(extract_feats(reg_path, "vis", vis_encoder)
            
            batch_vis[batch_idx, :, :] = np.concatenate(feat_list)
            batch_labels[batch_idx] =  1

            #neg samples
            curr_idx = randint(0, nb_captions)
            curr_len = len(captions[curr_idx])
            batch_text[batch_idx + 1, :curr_len] = captions[curr_idx]
            
            neg_idx = choice(range(curr_idx) + range(curr_idx+1, nb_samples))
            img_dir = join(crop_dir, str(neg_idx)) 
            reg_paths = [ join(img_dir, fname) for fname in listdir(img_dir)[:nb_reg] ]

            feat_list = []
            for reg_path in reg_paths:
                feat_list.append(extract_feats(reg_path, "vis", vis_encoder)
            
            batch_vis[batch_idx + 1, :, :] = np.concatenate(feat_list)
            batch_labels[batch_idx + 1] =  -1

        yield [batch_text, batch_vis], batch_labels

'''build the dataset for the experiments for the textual modality'''
def build_text_dataset(metadata, vocab, data_split, text_rep="word_level"):

    #extract the word2id dict
    word2id = vocab["word2id"]

    sequences = []
    #iterate over the items in the metadata
    for idx in trange(len(metadata)):
        
        #collect relevant metadata from the current item
        paragraph = metadata[idx]["paragraph"]
        
        if text_rep == "word_level":
            captions = sent_tokenize(paragraph)
            sequences.append(text_to_seq(captions, word2id))

    train_samples = int((1-data_split)*len(sequences))
    return {"captions": sequences[:train_samples]}, \
            {"captions": sequences[train_samples:]}             

'''build the dataset for the experiments for the visual modality'''
def build_vis_dataset(metadata):
   
    dest_dir = join(local_data_path, "crops")
    src_dir = join(local_data_path, "images")
   
    #make sure the dest dir exists
    if not exists(dest_dir):
        mkdir(dest_dir)

    for info in tqdm(metadata, total=len(metadata)):
        make_crops(str(info["id"]), info["regions"], src_dir, dest_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--op", help="operation to perform")
    parser.add_argument("--modality", default="text", help="the modality to work on")
    parser.add_argument("--vocab_lim", type=int, default=None, help="size of vocabulary to use")
    parser.add_argument("--data_split", default=0.2, type=float, help="ratio of train to validation samples")

    args = parser.parse_args()
    
    with open(join(local_data_path, METADICT_FNAME)) as f:
        meta_dict = json.load(f)

    if args.op == "build_dataset":
        
        if args.modality == "text":
            
            vocab_lim = "full" if args.vocab_lim is None else str(args.vocab_lim)
            vocab_fname = vocab_dict[str(vocab_lim)]
            assert exists(join(local_data_path, vocab_fname + ".p")), "build vocabulary first"
            vocab = load_pkl(local_data_path, vocab_fname)

            trainset, valset = build_text_dataset(meta_dict, vocab, args.data_split)
            train_caps_fname = "train_seq_" + str(args.vocab_lim)
            val_caps_fname = "val_seq_"  + str(args.vocab_lim)

            dump_pkl(trainset, local_data_path, train_caps_fname)
            dump_pkl(valset, local_data_path, val_caps_fname)

        elif args.modality == "vis":
            build_vis_dataset(metadata)
    
    elif args.op == "build_vocab":

        vocab = build_vocab(meta_dict, args.vocab_lim)
        fname = "text_vocab_full" if args.vocab_lim is None else "text_vocab_" + str(args.vocab_lim)
        dump_pkl(vocab, local_data_path, fname)
