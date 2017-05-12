###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Semantic Alignments
# Date Created: 5/8/2017
# 
# File Description: This script contains model definitions needed for the 
# experiments
###############################################################################

from __future__ import print_function
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Lambda, Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional
import pdb

from model_utils import calc_score, rank_svm_loss 

def text_model(vocab_size, max_len, embed_size, nb_hidden_states, batch_size, lr):
    
    input_seq = Input(batch_shape=(None, max_len))
    embedded = Embedding(vocab_size, embed_size, mask_zero=True)(input_seq)
    blstm = Bidirectional(LSTM(nb_hidden_states, return_sequences=True))(embedded)
    output_seq = TimeDistributed(Dense(vocab_size))(blstm)
    
    return output_seq

def full_model(vocab_size, max_len, embed_size, nb_hidden_states, 
                nb_regs, nb_feats=4096, common_dim=300, batch_size=32, lr=0.01):

    text_input = Input(batch_shape=(None, max_len))
    embedded = Embedding(vocab_size, embed_size, mask_zero=True)(text_input)
    blstm = Bidirectional(LSTM(nb_hidden_states, return_sequences=True))(embedded)
    text_out = TimeDistributed(Dense(common_dim, activation="relu"))(blstm)

    vis_input = Input(batch_shape=(None, nb_regs, nb_feats))
    vis_out = TimeDistributed(Dense(common_dim))(vis_input)
   
    score = Lambda(calc_score)([text_out, vis_out])

    model = Model(inputs=[text_input, vis_input], outputs=[score])
    print(model.summary())
    model.compile(loss=rank_svm_loss, optimizer="adam", class_mode="binary")

    return model
