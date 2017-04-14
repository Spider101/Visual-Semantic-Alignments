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
from os.path import join
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

