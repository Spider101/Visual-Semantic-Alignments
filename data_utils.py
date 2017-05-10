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

from text_utils import build_vocab, text_to_seq
from model_utils import encode_text
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

'''build the dataset for the experiments from the metadata dictionary'''
def build_text_dataset(metadata, vocab, data_split, text_rep="word_level"):

    #extract the word2id dict
    word2id = vocab["word2id"]

    #iterate over the items in the metadata
    for idx in trange(len(metadata)):
        
        #collect relevant metadata from the current item
        regions = metadata[idx]["regions"]
        paragraph = metadata[idx]["paragraph"]
        
        if text_rep == "word_level":
            captions = []
        
        #iterate over the regions corresponding to each item (image)
        for reg_idx in trange(len(regions)):

            caption = regions[reg_idx]["phrase"]
            
            #safety check to prevent blank strings being passed to the encoder
            if caption.replace(" ", "") == "":
                caption = "\b"

            captions.append(caption)
    
    sequences = text_to_seq(captions, word2id)
    train_samples = int((1-data_split)*len(sequences))
    return {"captions": sequences[:train_samples]}, {"captions": sequences[train_samples:]}             
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

    elif args.op == "build_vocab":

        vocab = build_vocab(meta_dict, args.vocab_lim)
        fname = "text_vocab_full" if args.vocab_lim is None else "text_vocab_" + str(args.vocab_lim)
        dump_pkl(vocab, local_data_path, fname)
