###############################################################################
# Author: Abhimanyu Banerjee
# Project: Visual Semantic Alignments
# Date Created: 4/12/2017
# 
# File Description: This script contains helper methods for manipulating the 
# textual component of the dataset
###############################################################################

from __future__ import print_function
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import *
import pdb

'''builds vocabulary from captions in dataset'''
def build_vocab(metadata, vocab_lim):

    sents = []
    for idx in trange(len(metadata)):

        #collect relevant metadata from the current item
        img_id = str(metadata[idx]["id"])
        regions = metadata[idx]["regions"]
        paragraph = metadata[idx]["paragraph"]

        sents += sent_tokenize(paragraph)

        #iterate over the regions corresponding to each item (image)
        for reg_idx in trange(len(regions)):

            #make sure the regions belong to the current image
            if int(img_id) != regions[reg_idx]["image_id"]:
                pdb.set_trace()

            caption = regions[reg_idx]["phrase"]
            
            #safety check to prevent blank strings being passed to the encoder
            if caption.replace(" ", "") == "":
                caption = "\b"

            sents.append(caption)

    word2id, id2word = tokenize_words(sents, vocab_lim)
    return {"word2id": word2id, "id2word": id2word}

'''converts words in sentences to tokens and returns dictionaries for translation
back and forth between words and the tokens'''
def tokenize_words(sents, vocab_lim):

    vocab_lim = len(sents) if vocab_lim is None else vocab_lim
    word2id, id2word, word_counts = {}, {}, {}
    sos_token, eos_token, unk_token = "<sos>", "<eos>", "<unk>"

    print("\nTokenizing words in the sentences..")
    for sent in sents:

        words = word_tokenize(sent)
        for word in words:
            
            if word not in word_counts:
                word_counts[word] = 0

            word_counts[word] += 1
    
    wcounts = list(word_counts.items())

    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    word2id = dict(list(zip(sorted_voc[:vocab_lim], list(range(3, vocab_lim-1)))))
    id2word = dict(list(zip(list(range(3, vocab_lim-1)), sorted_voc[:vocab_lim])))
    
    #add start of sentence token
    word2id[sos_token] = 2
    id2word[2] = sos_token

    #add end of sentence token
    word2id[eos_token] = 1
    id2word[1] = eos_token

    #add unknown word token
    word2id[unk_token] = vocab_lim-1
    id2word[vocab_lim-1] = unk_token

    print("Word tokenization complete! Vocabulary size is {}\n".format(len(word2id)))
    return word2id, id2word

'''converts sequences consisting of word indices back to sequences of words'''
def seq_to_texts(sequences, id2word):

    sents, flag = [], False
    for batch_idx in range(sequences.size(0)):
        
        words, seq = [], sequences[batch_idx]
        for seq_idx in range(seq.size(0)):
            
            word_id = seq[seq_idx].data[0]
            words.append(id2word[word_id])

            if word_id == 0:
                flag = True
                break

        sents.append(" ".join(words))
        
    return sents

'''convert words to word ids'''
def text_to_seq(sents, word2id):

    sos_token, eos_token, unk_token = "<sos>", "<eos>", "<unk>"
    word_ids = [ word2id[sos_token] ]
    
    for sent in sents:
        
        for word in word_tokenize(sent):

            if word not in word2id:
                word_ids.append(word2id[unk_token])
            else:
                word_ids.append(word2id[word])
    
    #tack the end of paragraph token at the end of the story and return it
    return word_ids + [ word2id[eos_token] ]

'''return the length of the longest sequence in a list of sequences'''
def get_max_len(seqs):

    return max([len(seq["text"]) for seq in seqs])
