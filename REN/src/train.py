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
import datetime


def train(epoch,batch_size, data,par,dr, _test):
    def val_test(d):
        _loss_val, _acc_val, _counter = 0.0, 0.0, 0
        for i, elem in enumerate(data.get_batch_train(batch_size,d)):
            dic = data.get_dic_val(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1])
            curr_loss_val, curr_acc_val = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
            _loss_val, _acc_val, _counter = _loss_val + curr_loss_val, _acc_val + curr_acc_val, _counter + 1
        logging.info("%s Loss: %.3f\t %s Accuracy: %.3f" % (d,_loss_val / float(_counter),d, _acc_val/float(_counter)))
        return _loss_val / float(_counter), _acc_val/float(_counter)

    def tr(verbose):
        loss, acc, counter = 0.0, 0.0, 0
        for i, elem in enumerate(data.get_batch_train(batch_size,'train')):
            dic = data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1],dr)
            curr_loss, curr_acc, _ = sess.run([entity_net.loss_val, entity_net.accuracy, entity_net.train_op],
                                              feed_dict=dic)
            loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
            if counter % verbose == 0:
                logging.info("Epoch %d\tBatch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f" % (e, counter, loss / float(counter), acc / float(counter)))
        return loss / float(counter), acc / float(counter)

    def init():
        # Setup Checkpoint + Log Paths
        ckpt_dir = "checkpoints/CNN{}/".format(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y").replace(":", "").replace(" ", ""))
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        # Initialize all Variables
        if os.path.exists(ckpt_dir + "checkpoint"):
            with open(ckpt_dir + "training_logs.pik", 'r') as f:
                train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = pickle.load(f)
        else:
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = {}, {}, {}, {}, {}, {}
        return ckpt_dir,train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

    tf.reset_default_graph()
    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)
        ckpt_dir,train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = init()

        # Initialize Saver
        saver = tf.train.Saver()
        # Initialize all Variables
        if os.path.exists(ckpt_dir + "checkpoint"):
            logging.info('Restoring Variables from Checkpoint!')
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            logging.info('Initializing Variables!')
            sess.run(tf.global_variables_initializer())

        # Get Current Epoch
        logging.info('Training stated')

        curr_epoch = sess.run(entity_net.epoch_step)
        best_val,patient= 0.0, 0
        for e in range(curr_epoch,epoch):

            train_loss[e], train_acc[e] = tr(10)
            val_loss[e], val_acc[e]     = val_test('val')
            if (_test):
                test_loss[e], test_acc[e] = val_test('test')
            # Update best_val
            if val_acc[e] >= best_val:
                best_val, patient = val_acc[e], 0
                with open(ckpt_dir + "training_logs.pik", 'w') as f:
                    pickle.dump((train_loss, train_acc, val_loss, val_acc), f)
            else:
                patient += 1

            # Early Stopping Condition
            if patient > 15:
                break
            sess.run(entity_net.epoch_increment)

        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, ckpt_dir
