###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Storytelling Experiments
# Date Created: 4/11/2017
#
# File Description: This script sets up the skip thought vectors module to be 
# used for encoding and decoding sentences as part of the project
###############################################################################

from __future__ import print_function
from os.path import join, exists
from os import mkdir, system
import shutil

from config.resources import (vist_data_path, skipthought_links)

def setup(dest_path):

    print("Setup commencing. Please wait ..\n")

    #remove any previous cache of the setup
    if exists(dest_path):
        
        print("Removing previous cache of skipthought embeddings. Please wait ..\n")
        shutil.rmtree(dest_path)

    mkdir(dest_path)

    #download all the required embedding files
    for idx in range(len(skipthought_links) - 1):
        
        system("wget -P {} {}".format(dest_path, skipthought_links[idx]))

    #download the skipthought script
    system("wget -P {} {}".format(join(".", "skip_thoughts"), skipthought_links[-1]))

    print("\nSetup Complete! Please fill in the skip thought script with the \
            appropriate paths to the embedding tables")
        
setup(join(vist_data_path, "skipthoughts"))
