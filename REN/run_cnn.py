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
            for i, elem in enumerate(data.get_batch_train(batch_size,'train')):
                dic = data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1],dr)
                curr_loss, curr_acc, _ = sess.run([entity_net.loss_val, entity_net.accuracy, entity_net.train_op],
                                                  feed_dict=dic)
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 10 == 0:
                    logging.info("Epoch %d\tBatch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f" % (e, counter, loss / float(counter), acc / float(counter)))
            # Add train loss, train acc to data
            train_loss[e], train_acc[e] = loss / float(counter), acc / float(counter)

            # Increment Epoch
            sess.run(entity_net.epoch_increment)


            if e % 1 == 0:
                val_loss_val, val_acc_val, counter_val = 0.0, 0.0, 0
                for i, elem in enumerate(data.get_batch_train(batch_size,'val')):
                    dic = data.get_dic_val(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1])
                    curr_loss_val, curr_acc_val = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
                    val_loss_val, val_acc_val, counter_val = val_loss_val + curr_loss_val, val_acc_val + curr_acc_val, counter_val + 1
                logging.info("Epoch %d Validation Loss: %.3f\tValidation Accuracy: %.3f" % (e, val_loss_val / float(counter_val), val_acc_val/float(counter_val)))
                # Add val loss, val acc to data
                val_loss[e], val_acc[e] = val_loss_val / float(counter_val), val_acc_val/float(counter_val)
                # Update best_val
                if val_acc[e] >= best_val:
                    best_val = val_acc[e]
                    patient = 0
                    # send_email("Epoch %d Validation Loss: %.3f\tValidation Accuracy: %.3f" % (e, val_loss_val, val_acc_val),'EX %s'%test_num)
                else:
                    patient += 1

                if(e%5 ==0):
                    # Save Model
                    saver.save(sess, ckpt_dir + "model.ckpt", global_step=entity_net.epoch_step)
                with open(ckpt_dir + "training_logs.pik", 'w') as f:
                    pickle.dump((train_loss, train_acc, val_loss, val_acc), f)


                # Test Loop
                test_loss_val, test_acc_val, counter_test = 0.0, 0.0, 0
                for i, elem in enumerate(data.get_batch_train(batch_size,'test')):
                    dic = data.get_dic_val(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1])
                    curr_loss_test, curr_acc_test = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
                    test_loss_val, test_acc_val, counter_test = test_loss_val + curr_loss_test, test_acc_val + curr_acc_test, counter_test + 1
                logging.info("Epoch %d Test Loss: %.3f\tTest Accuracy: %.3f" % (e, test_loss_val / float(counter_test), test_acc_val/float(counter_test)))

            # Early Stopping Condition
            if patient > 15:
                break

        # dic = data.get_dic_test(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob)
        # test_loss, test_acc = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
        #
        # # Print and Write Test Loss/Accuracy
        # logging.info("Test Loss: %.3f\tTest Accuracy: %.3f" % (test_loss, test_acc))
        saver.save(sess, ckpt_dir + "model.ckpt", global_step=entity_net.epoch_step)
        return train_loss, train_acc, val_loss, val_acc

def get_random_parameters(data,epoch,sent_len,sent_numb,embedding_size):
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """

    dists = dict(
    vocab_size = [data._data["vocab_size"]],
    label_num = [data._data["label_num"]],
    num_blocks = [5,10,20,50,70,90],
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

def main():
    embedding_size = 100
    epoch = 100
    sent_numb = 50
    sent_len = 30
    data = Dataset(train_size=10000,dev_size=None,test_size=None,sent_len=sent_len,
                    sent_numb=sent_numb, embedding_size=embedding_size)#, max_windows=None,win=None)

    batch_size_arr = [128,512,128,64,32]
    dr_arr = [0.2,0.5,0.7,0.9]
    best_accuracy = 0.0
    for exp in range(1,1000):
        batch_size = batch_size_arr[random.randint(0, len(batch_size_arr) - 1)]
        dr = dr_arr[random.randint(0, len(dr_arr) - 1)]
        par = get_random_parameters(data,epoch,sent_len,sent_numb,embedding_size)
        logging.info(par)
        train_loss, train_acc, val_loss, val_acc= train(epoch,batch_size, data,par,test_num='SMMQ_%s'%exp,dr=dr)
        par['id'] = 'Par'+ str(exp)
        par['acc'] = sorted([v for k,v in val_acc.items()])[-1]
        par['embeddings_mat']=''
        par['batch_size']= batch_size
        par['dr']= dr
        if (par['acc']>=best_accuracy):
            best_accuracy =par['acc']
            send_email("Best Accuracy: %.3f par: %s" % (best_accuracy,str(par)),'Sentence multi mask, with query. Ex: %s'%exp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main()
