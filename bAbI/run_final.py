from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import datetime
from src.utils.utils import Dataset, gen_embeddings
from src.EN import EntityNetwork
from src.trainer.train import train
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
from sklearn.grid_search import ParameterGrid
import cPickle as pickle
import logging
from random import shuffle

def get_parameters(data,epoch,sent_len,sent_numb,embedding_size, params):
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """
    embedding_file = 'data/glove.6B.{}d.txt'.format(embedding_size)
    embeddings_mat = gen_embeddings(data._data['word_idx'], embedding_size, embedding_file)
    dists = dict(
    vocab_size = data._data["vocab_size"],label_num = data._data["vocab_size"],
    sent_len = sent_len, sent_numb = sent_numb, embedding_size = embedding_size,
    embeddings_mat = embeddings_mat, clip_gradients= 40.0,
    max_norm = None, no_out = False, decay_steps = 0, decay_rate = 0, opt = 'Adam',
    num_blocks = params['nb'],
    learning_rate= params['lr'],
    trainable = params['tr'],
    L2 = params['L2']
    )
    return dists

def main(task_num,sample_size=''):
    embedding_size = 100
    epoch = 300
    best_accuracy = 0.0
    grind_ris={}

    if not os.path.exists('data/ris/task_{}'.format(task_num)):
        os.makedirs('data/ris/task_{}'.format(task_num))

    param_grid = {'nb': [40],
                  'lr': [0.001],
                  'tr': [[0,0,0,0]],
                  'L2': [0.001],# [0.0,0.1,0.01,0.001,0.0001]
                  'bz': [32],
                  'dr': [0.7],
                  }
    grid = list(ParameterGrid(param_grid))
    np.random.shuffle(grid)
    for params in list(grid):
        data = Dataset('data/tasks_1-20_v1-2/en-valid{}/'.format(sample_size),int(task_num))

        ## for sentence
        par = get_parameters(data,epoch,data._data['sent_len'],data._data['sent_numb'],embedding_size,params)
        t = train(epoch,params['bz'], data, par, dr=params['dr'], _test=True)

        acc = sorted([v for k,v in t[5].items()])[-1]

        if (acc > best_accuracy):
            best_accuracy = acc

        grind_ris[str(params)] = acc

        f_save = 'data/ris/task_{}/{}.PIK'.format(task_num,str(params)+str(acc))
        with open(f_save, 'w') as f:
            pickle.dump((t), f)



    # batch_size = 32
    # epoch = 200
    # if not os.path.exists('data/ris/task_{}'.format(task_num)):
    #     os.makedirs('data/ris/task_{}'.format(task_num))
    # data = Dataset('data/tasks_1-20_v1-2/en-valid{}/'.format(sample_size),int(task_num))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main(task_num=sys.argv[1])
