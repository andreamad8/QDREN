from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time
import datetime
from src.ulilz_CNN import Dataset
from src.EN import EntityNetwork
from src.train import train
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import cPickle as pickle
import logging
from data.email_util import send_email



def get_random_parameters(data,epoch,sent_len,sent_numb,embedding_size):
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """

    dists = dict(
    vocab_size = [data._data["vocab_size"]],
    label_num = [data._data["label_num"]],
    num_blocks = [5],#,10,20,50,70,90],
    sent_len = [sent_len],
    sent_numb = [sent_numb],
    embedding_size = [embedding_size],
    embeddings_mat = [data._data["embeddings_mat"]],
    learning_rate= [0.1,0.01,0.001,0.0001],
    clip_gradients= [-10.0,-1.0,1.0,10.0,40.0],
    opt = ['Adam','RMSProp'],
    trainable = [[1,1,0,0],[1,0,0,0],[0,0,0,0]],
    max_norm = [None],#[None,None,1],
    no_out = [False],
    decay_steps = [0], #  [epoch* data._data['len_training']/25,epoch* data._data['len_training']/100],
    decay_rate = [0],
    L2 = [0.001,0,0.01,0.001,0.0001,0.00001]
    )
    d = {}
    for k in dists:
        if hasattr(dists[k], '__len__'):
            idx = random.randint(0, len(dists[k]) - 1)
            d[k] = dists[k][idx]
        else:
            d[k] = dists[k].rvs()
    return d


def get_par_arr(a):
    return a[random.randint(0, len(a) - 1)]

def main():
    embedding_size = 50
    epoch = 1
    sent_numb,sent_len = 50,30
    max_windows,win = None,None
    data = Dataset(train_size=10,dev_size=1,test_size=1,sent_len=sent_len,
                    sent_numb=sent_numb, embedding_size=embedding_size,
                    max_windows=max_windows,win=win)

    best_accuracy = 0.0
    for exp in range(1,2):
        batch_size = get_par_arr([128,512,128,64,32])
        dr = get_par_arr([0.2,0.5,0.7,0.9])
        ## for sentence
        par = get_random_parameters(data,epoch,sent_len,sent_numb,embedding_size)
        ## for windows
        # par = get_random_parameters(data,epoch,(win*2)+1,max_windows,embedding_size)
        logging.info(par)
        t = train(epoch,batch_size, data, par, dr=dr, _test=False)
        par['acc'] = sorted([v for k,v in t[3].items()])[-1]
        par['embeddings_mat'],par['batch_size'],par['dr']= '' ,batch_size,dr
        if (par['acc']>=best_accuracy):
            best_accuracy =par['acc']
            send_email("Best Accuracy: %.3f par: %s" % (best_accuracy,str(par)),'%s in' % str(t[6]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main()
