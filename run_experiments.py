###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Semantic Alignments
# Date Created: 5/10/2017
# 
# File Description: This script contains code to run the experiments for this 
# project and evaluate performance
###############################################################################

from __future__ import print_function
import argparse
import os
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.INFO)
import pdb
from tqdm import *

from config.resources import (trainset_dict, valset_dict, local_data_path)
from data_utils import data_gen
from text_utils import get_max_len
from utils import load_pkl, dump_pkl
from model_zoo import full_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def build_data_loader(mode, dataset, batch_size, nb_regions, vocab_type, vocab_lim):

    if dataset == "multimodal":
        
        if mode == "train":

            train_fname = trainset_dict[vocab_type][str(vocab_lim)]
            train_text = load_pkl(local_data_path, train_fname)
            train_max_len = get_max_len(train_text["captions"])
            train_batches = int(len(train_text["captions"]) / batch_size)
                        
            val_fname = valset_dict[vocab_type][str(vocab_lim)]
            val_text = load_pkl(local_data_path, val_fname)
            val_max_len = get_max_len(val_text["captions"])
            val_batches = int(len(val_text["captions"]) / batch_size)

            max_len = max(train_max_len, val_max_len)
            train_loader = data_gen(train_text, max_len, nb_reg=nb_regions, 
                                    batch_size=batch_size)
            val_loader = data_gen(val_text, max_len, nb_reg=nb_regions, 
                                    batch_size=batch_size)

    return (train_max_len, val_max_len), (train_batches, val_batches), \
            (train_loader, val_loader)

def train_model(model, data_loader, batch_size, nb_batches, nb_epochs):
    
    train_loader, val_loader = data_loader
    train_iters, val_iters = nb_batches
   
    for epoch in trange(nb_epochs):

        print("Epoch: ", epoch + 1)
        iters, loss = 0, 0
        for inputs, targets in tqdm(train_loader, total=train_iters):
            
            loss += model.train_on_batch(inputs, targets)
            iters += 1
            if iters % 50 == 49:
                print("Loss at iteration {} is {}".format(iters+1, 
                                            loss / ((iters+1)*batch_size) ))

            if iters >= train_iters:
                break

        iters, val_loss = 0, 0
        for inputs, targets in tqdm(val_loader, total=val_iters):
            
            pdb.set_trace()
            val_loss += model.test_on_batch(inputs, targets)
            iters += 1

            if iters >= val_iters:
                break

        print("\nValidation loss at the end of epoch {} is {}".format(epoch+1, 
                                            val_loss/(val_iters*batch_size)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="train", help="what operation to perform")
    parser.add_argument("--dataset", default="multimodal", help="which dataset to use")
    parser.add_argument("--vocab_type", default="freq")
    parser.add_argument("--debug", action="store_true", help="debug or not")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("--vocab_size", type=int, default=1000, help="size of the vocabulary")
    parser.add_argument("--embed_size", type=int, default=300, help="size of textual embedding layer")
    parser.add_argument("--hidden_size", type=int, default=512, help="number of hidden states in rnn")
    parser.add_argument("--nb_regions", type=int, default=10, help="number of regions in an number")
    parser.add_argument("--batch_size", type=int, default=32, help="number of samples in a batch")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="ratio of training to test samples")

    args = parser.parse_args()

    if args.op == "train":
        
        max_seq_len, nb_batches, data_loader = build_data_loader(args.op, 
                                                args.dataset, args.batch_size, 
                                                args.nb_regions, args.vocab_type, 
                                                args.vocab_size)

        model = full_model(args.vocab_size, max(max_seq_len), args.embed_size, 
                            args.hidden_size, args.nb_regions, 
                            batch_size=args.batch_size, lr=args.lr)

        train_model(model, data_loader, args.batch_size, nb_batches, args.epochs)

