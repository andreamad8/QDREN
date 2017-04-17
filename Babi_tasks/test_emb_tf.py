import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import gensim
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
import codecs
import sys
from tqdm import tqdm
import cPickle as pickle
from src.utils import Dataset

def get_vocab_and_index(emb_file='data/glove.6B.100d.txt'):
    f = codecs.open(emb_file, "r", "utf-8")
    # load matrix
    embeddings_index = {}
    vocab = []
    embeddings_index["<UNK>".decode('utf-8')]= np.zeros(100)
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        vocab.append(word)
        coefs = np.asarray(values[1:]).astype(np.float)
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index

data= Dataset('data/tasks_1-20_v1-2/en-valid/',1)
print(data._data['word_idx'])
print(data._data['vocab_size'])

embeddings_index = get_vocab_and_index()
embed_size = 100
embedding_matrix = np.zeros((data._data['vocab_size'], embed_size))
for word, i in data._data['word_idx'].items():
    if word in embeddings_index:
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            print('Missing from QAEMB: {}'.format(word))

    else:
        print('Missing from QAEMB: {}'.format(word))
print('Total number of null word embeddings:')
print(np.sum(np.sum(embedding_matrix, axis=1) == 0))





# get_vocab_and_index()






# embed_size = 100
# VOCAB =
# model =  gensim.models.Word2Vec.load('data/glove.6B.100d.txt')
# # prepare embedding matrix
# embedding_matrix = np.zeros((VOCAB, embed_size))
# for word, i in tokenizer.word_index.items():
#     if word in model:
#         embedding_vector = model[word]
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#     #else:
#     #    print('Missing from QAEMB: {}'.format(word))
#
# print('Total number of null word embeddings:')
# print(np.sum(np.sum(embedding_matrix, axis=1) == 0))
