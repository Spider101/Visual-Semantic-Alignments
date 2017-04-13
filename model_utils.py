###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Semantic Alignments
# Date Created: 4/12/2017
#
# File Description: This script contains helper methods related to the different
# architectures to be used for various experiments
###############################################################################

from __future__ import print_function
import pdb

'''encode sentence(s) to a fixed vector representation using the specifier 
encoder'''
def encode_text(encoder, text):

    #make input text is in a list (even if its a singleton list
    if type(text) is not list:
        text = [text]

    return encoder.encode(text)
