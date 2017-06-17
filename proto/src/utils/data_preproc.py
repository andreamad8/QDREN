import numpy as np
import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
from iteration_utilities import flatten
import os
def parse_file(file_name):
    f = open(file_name, 'r')
    line = f.readline()
    line = f.readline()

    ##paragraph
    line = f.readline()
    s = line.split(' ')
    s[-1] = s[-1].rstrip()

    line = f.readline()

    ### question
    line = f.readline()
    q = line.split(' ')
    q[-1] = q[-1].rstrip()

    line = f.readline()

    ### answer
    a = f.readline()
    a = a.rstrip()


    line = f.readline()
    entity = {}
    while True:
        line = f.readline()
        if not line:
            break
        t = line.split(':')
        entity[t[0]]=t[1][:-1]

    for i, w in enumerate(s):
        if len(w)>5 and w[0]=='@':
            s[i] = entity[w]

    for i, w in enumerate(q):
        if w == '@placeholder':
            q[i] = entity[a]
        elif len(w)>5 and w[0]=='@':
            q[i] = entity[w]

    s = '\t'.join(s)
    q = '\t'.join(q)
    a = entity[a]
    return q + '\n' + a + '\n' + s + '\n\n'

data = ''
for filename in os.listdir('data/story/'):
    print('data/story/{}'.format(filename))
    data += parse_file('data/story/{}'.format(filename))

with open('data/train.txt', 'w') as f:
    f.write(data)
#
# from nltk import ne_chunk, pos_tag, word_tokenize
# from nltk.tree import Tree
#
# def get_continuous_chunks(text):
#     chunked = ne_chunk(pos_tag(word_tokenize(text)))
#     prev = None
#     continuous_chunk = []
#     current_chunk = []
#     for i in chunked:
#         if type(i) == Tree:
#             current_chunk.append(" ".join([token for token, pos in i.leaves()]))
#         elif current_chunk:
#             named_entity = " ".join(current_chunk)
#             if named_entity not in continuous_chunk:
#                 continuous_chunk.append(named_entity)
#                 current_chunk = []
#         else:
#             continue
#     return continuous_chunk
#     print(get_continuous_chunks(line))
