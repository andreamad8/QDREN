from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time
import datetime
from src.utils.utils_CBT import Dataset
from src.EN import EntityNetwork
from src.trainer.train_cbt import train
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import cPickle as pickle
import logging
from data.email_util import send_email
from sklearn.grid_search import ParameterGrid
import cPickle as pickle
from random import shuffle

def get_parameters(data,epoch,sent_len,sent_numb,embedding_size, params):
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """

    dists = dict(
    vocab_size = data._data["vocab_size"],label_num = data._data["label_num"],
    sent_len = sent_len, sent_numb = sent_numb, embedding_size = embedding_size,
    embeddings_mat = data._data["embeddings_mat"], clip_gradients= 40.0,
    max_norm = None, no_out = False, decay_steps = 0, decay_rate = 0, opt = 'Adam',
    num_blocks = params['nb'],
    learning_rate= params['lr'],
    trainable = params['tr'],
    L2 = params['L2']
    )
    return dists



# def get_random_parameters(data,epoch,sent_len,sent_numb,embedding_size):
#     """
#     create a random configuration from using dists to select random parameters
#     :return: neural network parameters for create_model
#     """
#
#     dists = dict(
#     vocab_size = [data._data["vocab_size"]],
#     label_num = [data._data["label_num"]],
#     num_blocks = [10],# [10,20,50],
#     sent_len = [sent_len],
#     sent_numb = [sent_numb],
#     embedding_size = [embedding_size],
#     embeddings_mat = [data._data["embeddings_mat"]],
#     learning_rate= [0.001,0.0001], #0.1,0.01,0.001,
#     clip_gradients= [40.0],#[-10.0,-1.0,1.0,10.0,40.0],
#     opt = ['Adam'],#['Adam','RMSProp'],
#     trainable = [[1,0,0,0]],#[1,1,0,0],[1,0,0,0]],#[[1,1,0,0],[1,0,0,0],[0,0,0,0]],
#     max_norm = [None],#[None,None,1],
#     no_out = [False],
#     decay_steps = [0], #  [epoch* data._data['len_training']/25,epoch* data._data['len_training']/100],
#     decay_rate = [0],
#     L2 = [0.0]# [0.0,0.1,0.01,0.001,0.0001]
#     )
#     d = {}
#     for k in dists:
#         if hasattr(dists[k], '__len__'):
#             idx = random.randint(0, len(dists[k]) - 1)
#             d[k] = dists[k][idx]
#         else:
#             d[k] = dists[k].rvs()
#     return d
#
#
# def get_par_arr(a):
#     return a[random.randint(0, len(a) - 1)]

def main():
    embedding_size = 100
    max_windows = 100
    epoch = 2
    best_accuracy = 0.0
    grind_ris={}
    file_grid = "data/grid/grid_simple_CBT_NE_SIMPLE.pik"
    param_grid = {'nb': [10],
                  'lr': [0.001,0.0001],
                  'tr': [[1,1,0,0],[1,0,0,0],[0,0,0,0]],
                  'L2': [0.0],# [0.0,0.1,0.01,0.001,0.0001]
                  'w' : [3,4,5],
                  'bz': [32,64],
                  'dr': [1.0],
                  }
    grid = list(ParameterGrid(param_grid))
    np.random.shuffle(grid)
    for params in list(grid):
        data = Dataset(train_size=1,dev_size=1,test_size=1,
                        sent_len=None,sent_numb=None, embedding_size=embedding_size,
                        max_windows=max_windows,win=params['w'], ty_CN_NE ='NE')

        ## for sentence
        # par = get_random_parameters(data,epoch,sent_len,sent_numb,embedding_size)
        ## for windows

        par = get_parameters(data,epoch,(params['w']*2)+1,
                            max_windows,embedding_size, params)

        t = train(epoch,params['bz'], data, par, dr=params['w'], _test=False)

        acc = sorted([v for k,v in t[3].items()])[-1]

        if (acc > best_accuracy):
            best_accuracy = acc
            with open(file_grid, 'w') as f:
                pickle.dump((grind_ris), f)
            send_email("CBT NE Best Accuracy SIMPLE: %.3f \t PAR: %s" % (best_accuracy,str(params)),'in %s' % str(t[6]))

        grind_ris[str(params)] = acc

    with open(file_grid, 'w') as f:
        pickle.dump((grind_ris), f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main()
    # max_windows,win = get_par_arr([100]),get_par_arr([3])#,3,4,5])
    # # sent_numb,sent_len = get_par_arr([20]),get_par_arr([50])#,3,4,5])
