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
import argparse

from config.resources import (local_data_path, skipthought_links)

def setup_encoder(weights_path, scripts_path):

    print("Setup commencing. Please wait ..\n")

    #remove any previous cache of the setup
    if exists(weights_path):
        
        print("Removing previous cache of skipthought embeddings. Please wait ..\n")
        shutil.rmtree(weights_path)

    mkdir(weights_path)

    encoder_links = skipthought_links["encoder"]

    #download all the required embedding files
    for idx in range(len(encoder_links) - 1):
        
        system("wget -P {} {}".format(weights_path, encoder_links[idx]))

    #download the skipthought script
    system("wget -P {} {}".format(scripts_path, encoder_links[-1]))

    print("\nSetup Complete! Please fill in the skip thought script with the \
            appropriate paths to the embedding tables")

def setup_decoder(weights_path, scripts_path):

    print("Setup commencing. Please wait ..\n")

    #check if encoder has been setup
    assert exists(weights_path) and exists(scripts_path), "please setup the encoder first"

    decoder_links = skipthought_links["decoder"]

    #download all the required embedding files
    for idx in range(len(decoder_links)):
        
        system("wget -P {} {}".format(scripts_path, decoder_links[idx]))

    print("\nSetup Complete!") 
       
if __name__ == "__main__":
    
    weights_path = join(local_data_path, "skipthoughts")
    scripts_path = join(".", "skip_thoughts")

    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder", action="store_true", help="set up encoder or not")
    parser.add_argument("--decoder", action="store_true", help="set up decoder or not")

    args = parser.parse_args()

    if args.encoder:
       setup_encoder(weights_path, scripts_path)

    elif args.decoder:
        setup_decoder(weights_path, scripts_path)
        
