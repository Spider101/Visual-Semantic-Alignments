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
from os.path import join, exists
from os import listdir, mkdir
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np
import pdb
from tqdm import *

from config.resources import (trainset_dict, valset_dict, local_data_path, 
                                MODEL_CHKPNT_FNAME)
from config.push_notifs import send_message
from data_utils import data_gen
from text_utils import get_max_len
from utils import *
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

def experiment_setup():

    #get unique id for experiment
    exp_id = get_unique_hash()
            
    #check if logs directory exists
    log_dir = join(local_data_path, "logs")
    if not exists(log_dir):
        mkdir(log_dir)

    log_path = join(log_dir, exp_id)
    mkdir(log_path)

    if args.checkpoint:
        
        #check if checkpoint directory exists
        chkpnt_dir = join(local_data_path, "checkpoints")
        if not exists(chkpnt_dir):
            mkdir(chkpnt_dir)
        
        chkpnt_path = join(chkpnt_dir, exp_id)
        mkdir(chkpnt_path)
    else:
        chkpnt_path = None
    
    return log_path, chkpnt_path, exp_id


def train_model(model, data_loader, config):
    
    #unpack config object
    batch_size, nb_batches, nb_epochs = config["batch_size"], config["nb_batches"], \
                                        config["nb_epochs"]
    log_path, chkpnt_path = config["log_path"], config["chkpnt_path"]

    train_loader, val_loader = data_loader
    train_iters, val_iters = nb_batches
    log_freq = int(train_iters / 5)
  
    if log_path:
        train_metric, val_metric = [], []

    for epoch in range(nb_epochs):

        print("Epoch: ", epoch + 1)
        iters, loss = 0, 0
        for inputs, targets in tqdm(train_loader, total=train_iters):
            
            loss += model.train_on_batch(inputs, targets)
            iters += 1
            if iters % log_freq == (log_freq - 1):
                print("Loss at iteration {} is {}".format(iters+1, 
                                            loss / ((iters+1)*batch_size) ))

            if iters >= train_iters:
                break

        iters, best_metric, val_loss = 0, np.inf, 0.0
        for inputs, targets in tqdm(val_loader, total=val_iters):
            
            val_loss += model.test_on_batch(inputs, targets)
            iters += 1

            if iters >= val_iters:
                break

        print("\nValidation loss at the end of epoch {} is {}".format(epoch+1, 
                                            val_loss/(val_iters*batch_size)))
        
        if log_path:
            
            #log metrics at the end of the epoch
            train_metric.append(loss/(train_iters*batch_size))
            val_metric.append(val_loss/(val_iters*batch_size))

            epoch_list = [i for i in range(nb_epochs+1)]
            plot_path = join(log_path, "training.png")
            #TODO: plot regularly

            dump_pkl({"x": [epoch_list, epoch_list], "y": [train_metric, val_metric]},
                    log_path, "training_data")

        if chkpnt_path:
            
            #update best validation loss (metric of choice for now)
            if best_metric > val_loss/(val_iters*batch_size):
                
                best_metric = val_loss/(val_iters*batch_size)
                
                #checkpoint the model
                weights_fname = MODEL_CHKPNT_FNAME.format(epoch, best_metric)
                model.save_weights(join(chkpnt_path, weights_fname))
   


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="train", help="what operation to perform")
    parser.add_argument("--dataset", default="multimodal", help="which dataset to use")
    parser.add_argument("--vocab_type", default="freq")
    parser.add_argument("--checkpoint", action="store_true", help="checkpoint or not")
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

        if args.debug:
            log_path, chkpnt_path = None, None
        
        else:
            log_path, chkpnt_path, exp_id = experiment_setup()
        
        #pack all configuration details into a config object
        config = {"batch_size": args.batch_size, "nb_batches": nb_batches, \
                    "nb_epochs": args.epochs, "log_path": log_path, \
                    "chkpnt_path": chkpnt_path}

        train_model(model, data_loader, config)
        subject = "Training for #{} completed".format(exp_id)
        messg = str(vars(args))
        send_message(subject, body=messg)
