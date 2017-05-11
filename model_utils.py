###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Semantic Alignments
# Date Created: 4/12/2017
#
# File Description: This script contains helper methods related to the different
# architectures to be used for various experiments
###############################################################################

from __future__ import print_function
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.backend as K
import numpy as np
import pdb

def build_feat_extractor(modality):
    
    if modality == "vis":
        
        model = VGG16()
        model.layers.pop()
        model.layers.pop()
        model.outputs = [ model.layers[-1].output ]
        model.layers[-1].outbound_nodes = []

        return model

def extract_feats(data_path, modality, extractor, verbosity=False):

    if modality == "vis":
        
        img = image.load_img(data_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        return extractor.predict(x, verbose=verbosity)

'''encode sentence(s) to a fixed vector representation using the specifier 
encoder'''
def encode_text(encoder, text):

    #make input text is in a list (even if its a singleton list
    if type(text) is not list:
        text = [text]

    return encoder.encode(text)

def calc_score(t1, t2):

    t2 = K.permute_dimensions(t2, (0, 2, 1))
    outer_prod = K.batch_dot(t1, t2)
    
    return K.sum(K.maximum(outer_prod, axis=1))

def rank_svm_loss(ytrue, ypred, margin=1):

    signed = y_pred * y_true 
    pos = signed[0]
    neg = signed[1]
    rank_hinge_loss = K.mean( K.relu( margin - pos - neg ), axis=-1)
    return rank_hinge_loss
