###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Semantic Alignments
# Date Created: 4/14/2017
# 
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments 
###############################################################################

from __future__ import print_function
import numpy as np
from pickle import load, dump
import matplotlib.pyplot as plt
from os.path import join, getctime, exists
from os import listdir
import time
import hashlib
import pdb

'''wrapper method to load pickle files and return the data stored in them'''
def load_pkl(data_dir, pklName, verbose=True):
    
    if verbose:
        print("\nLoading data from {0}/{1}.p".format(data_dir, pklName))
                    
    return load(open(join(data_dir, pklName + '.p'), 'rb'))

'''wrapper method to save any and all data to pickle files'''
def dump_pkl(data, data_dir, pklName, verbose = True):

    if verbose:
        print("\nDumping data into {0}/{1}.p".format(data_dir, pklName))

    dump(data, open(join(data_dir, pklName + '.p'), 'wb'))

def get_last_chkpnt(chkpnt_path):
    
    assert exists(chkpnt_path), \
            "Sorry, there are no checkpoints available at {}".format(chkpnt_path)
    checkpoints = [join(chkpnt_path, fname) for fname in listdir(chkpnt_path)]
    return max(checkpoints, key=getctime)

def get_unique_hash():

    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    #return first 10 characters of the hash (too long otherwise)
    return hash.hexdigest()[:10]

def plot_lines(x_list, y_list, legend, xlab, ylab, title=None):
    
    assert isinstance(legend, tuple), "Legend to plot must be a tuple"

    for idx in range(len(legend)):
        plt.plot(x_list[idx], y_list[idx], label=legend[idx])
    
    plt.xlabel(xlab) 
    plt.ylabel(ylab)
    plt.legend()

    if title:
        plt.title(title)
    
    plt.show()
    plt.gcf().clear()

