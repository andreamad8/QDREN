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
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import cPickle as pickle
import logging
from data.email_util import send_email

def train(epoch,batch_size, data,par,test_num,dr):

    tf.reset_default_graph()
    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)


        logging.info('Initializing Variables!')
        sess.run(tf.global_variables_initializer())
        train_loss, train_acc, val_loss, val_acc = {}, {}, {}, {}

        # Get Current Epoch
        curr_epoch = sess.run(entity_net.epoch_step)

        logging.info('Training stated')
        loss, acc, counter = 0.0, 0.0, 0
        for i, elem in enumerate(data.get_batch_train(batch_size,'train')):
            dic = data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1],dr)
            le,curr_loss, curr_acc, _ = sess.run([entity_net.length,entity_net.loss_val, entity_net.accuracy, entity_net.train_op],
                                              feed_dict=dic)
            logging.info(le)

        sess.run(entity_net.epoch_increment)


        return train_loss, train_acc

def get_random_parameters(data,epoch,sent_len,sent_numb,embedding_size):
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """

    dists = dict(
    vocab_size = [data._data["vocab_size"]],
    label_num = [data._data["label_num"]],
    num_blocks = [5],
    sent_len = [sent_len],
    sent_numb = [sent_numb],
    embedding_size = [embedding_size],
    embeddings_mat = [data._data["embeddings_mat"]],
    learning_rate= [0.1],
    clip_gradients= [1.0],
    opt = ['Adam'],
    trainable = [[1,1,0,0]],
    max_norm = [None],#[None,None,1],
    no_out = [False],
    decay_steps = [0], #  [epoch* data._data['len_training']/25,epoch* data._data['len_training']/100],
    decay_rate = [0],
    L2 = [0.0001]
    )


    d = {}
    for k in dists:
        if hasattr(dists[k], '__len__'):
            idx = random.randint(0, len(dists[k]) - 1)
            d[k] = dists[k][idx]
        else:
            d[k] = dists[k].rvs()
    return d

def main():
    embedding_size = 50
    epoch = 100
    sent_numb ,sent_len = None, None
    max_windows,win = 64 , 3
    data = Dataset(train_size=100,dev_size=1,test_size=1,sent_len=sent_len,
                    sent_numb=sent_numb, embedding_size=embedding_size,
                    max_windows=max_windows,win=win)

    batch_size_arr = [1]
    dr_arr = [0.7]
    best_accuracy = 0.0
    batch_size = batch_size_arr[random.randint(0, len(batch_size_arr) - 1)]
    dr = dr_arr[random.randint(0, len(dr_arr) - 1)]
    par = get_random_parameters(data,epoch,(win*2)+1,max_windows,embedding_size)
    train_loss, train_acc= train(epoch,batch_size, data,par,test_num=1,dr=dr)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main()
