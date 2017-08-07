from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time
import datetime
from src.utils.ulilz_CNN import Dataset
from src.trainer.train import train
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import cPickle as pickle
import logging
from data.email_util import send_email



def get_parameters(data,epoch,sent_len,sent_numb,embedding_size):
    """
    create final conf
    :return: neural network parameters for create_model
    """

    dists = dict(
    vocab_size = data._data["vocab_size"],
    label_num = data._data["label_num"],
    num_blocks = 20,
    sent_len = sent_len,
    sent_numb = sent_numb,
    embedding_size = embedding_size,
    embeddings_mat = data._data["embeddings_mat"],
    learning_rate= 0.01,
    clip_gradients= 40.0,
    opt = 'RMSProp',
    trainable = [1,1,0,0],
    max_norm = None,
    no_out = False,
    decay_steps = 0,
    decay_rate = 0,
    L2 = 0.0001
    )
    return dists

#td.group_by_batches(shuffled, batch_size):

def main():
    embedding_size = 100
    epoch = 200
    sent_numb,sent_len =  None,None
    max_windows,win = 150,4
    batch_size = 64
    dr = 0.5
    data = Dataset(train_size=None,dev_size=None,test_size=None,sent_len=sent_len,
                    sent_numb=sent_numb, embedding_size=embedding_size,
                    max_windows=max_windows,win=win)

    # ## for sentence
    # par = get_parameters(data,epoch,sent_len,sent_numb,embedding_size)
    # ## for windows
    par = get_parameters(data,epoch,(win*2)+1,max_windows,embedding_size)
    logging.info(par)
    t = train(epoch,batch_size, data, par, dr=dr, _test=True)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main()
