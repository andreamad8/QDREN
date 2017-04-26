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

def train(epoch,batch_size, data,par,test_num):
    # Setup Checkpoint + Log Paths
    ckpt_dir = "checkpoints/CNN{}/".format(test_num)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    tf.reset_default_graph()
    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)

        # Initialize Saver
        saver = tf.train.Saver()

        # Initialize all Variables
        if os.path.exists(ckpt_dir + "checkpoint"):
            logging.info('Restoring Variables from Checkpoint!')
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            with open(ckpt_dir + "training_logs.pik", 'r') as f:
                train_loss, train_acc, val_loss, val_acc = pickle.load(f)
        else:
            logging.info('Initializing Variables!')
            sess.run(tf.global_variables_initializer())
            train_loss, train_acc, val_loss, val_acc = {}, {}, {}, {}

        # Get Current Epoch
        curr_epoch = sess.run(entity_net.epoch_step)

        logging.info('Training stated')
        best_val= 0.0
        patient = 0
        for e in range(curr_epoch,epoch):
            loss, acc, counter = 0.0, 0.0, 0
            for i, elem in enumerate(data.get_batch_train(batch_size)):
                dic = data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1])
                curr_loss, curr_acc, _ = sess.run([entity_net.loss_val, entity_net.accuracy, entity_net.train_op],
                                                  feed_dict=dic)
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 1 == 0:
                    logging.info("Epoch %d\tBatch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f" % (e, counter, loss / float(counter), acc / float(counter)))
            # Add train loss, train acc to data
            train_loss[e], train_acc[e] = loss / float(counter), acc / float(counter)

            # Increment Epoch
            sess.run(entity_net.epoch_increment)


            if e % 1 == 0:
                dic = data.get_dic_val(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob)
                val_loss_val, val_acc_val = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
                logging.info("Epoch %d Validation Loss: %.3f\tValidation Accuracy: %.3f" % (e, val_loss_val, val_acc_val))
                send_email("Epoch %d Validation Loss: %.3f\tValidation Accuracy: %.3f" % (e, val_loss_val, val_acc_val),'EX %s'%test_num)
                # Add val loss, val acc to data
                val_loss[e], val_acc[e] = val_loss_val, val_acc_val

                # Update best_val
                if val_acc[e] >= best_val:
                    best_val = val_acc[e]
                else:
                    patient += 1

                # Save Model
                saver.save(sess, ckpt_dir + "model.ckpt", global_step=entity_net.epoch_step)
                with open(ckpt_dir + "training_logs.pik", 'w') as f:
                    pickle.dump((train_loss, train_acc, val_loss, val_acc), f)

            # Early Stopping Condition
            if patient > 2:
                break
        # Test Loop
        dic = data.get_dic_test(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob)
        test_loss, test_acc = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)

        # Print and Write Test Loss/Accuracy
        logging.info("Test Loss: %.3f\tTest Accuracy: %.3f" % (test_loss, test_acc))
        return train_loss, train_acc, val_loss, val_acc,test_loss, test_acc



def main():
    embedding_size = 100
    sent_len = 2
    sent_numb = 4
    batch_size = 32
    epoch = 10
    dr = 0.5
    data = Dataset(train_size=100,dev_size=1,test_size=1,sent_len=sent_len,
                    sent_numb=sent_numb, embedding_size=embedding_size, dr= dr)

    par = dict(
    vocab_size = data._data["vocab_size"],
    label_num = data._data["label_num"],
    num_blocks = data._data["label_num"],
    sent_len = sent_len,
    sent_numb = sent_numb,
    embedding_size = embedding_size,
    embeddings_mat = data._data["embeddings_mat"],
    learning_rate=  0.001,
    clip_gradients= 40.0,
    opt = 'Adam', # 'RMSProp', 'SGD'
    trainable = [1,1,0,0],
    max_norm = None,
    no_out = True,
    decay_steps = 20, #  [epoch* data._data['len_training']/25,epoch* data._data['len_training']/100],
    decay_rate = 0.5,
    L2 = 0.1
    )


    train(epoch,batch_size, data,par,test_num=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main()
