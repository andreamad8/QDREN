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


def train(epoch,batch_size, data,par):
    tf.reset_default_graph()
    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)
        logging.info('Init stated')
        sess.run(tf.global_variables_initializer())
        train_loss, train_acc, val_loss, val_acc = {}, {}, {}, {}

        logging.info('Training stated')
        best_val=0.0
        for e in range(epoch):
            loss, acc, counter = 0.0, 0.0, 0
            for i, elem in enumerate(data.get_batch_train(batch_size)):
                dic = data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,elem[0],elem[1])
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
                dic = data.get_dic_val(entity_net.S,entity_net.Q,entity_net.A)
                val_loss_val, val_acc_val = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
                logging.info("Epoch %d Validation Loss: %.3f\tValidation Accuracy: %.3f" % (e, val_loss_val, val_acc_val))

                # Add val loss, val acc to data
                val_loss[e], val_acc[e] = val_loss_val, val_acc_val

                # Update best_val
                if val_acc[e] > best_val:
                    best_val = val_acc[e]

                # # Save Model
                # saver.save(sess, ckpt_dir + "model.ckpt", global_step=entity_net.epoch_step)
                # with open(ckpt_dir + "training_logs.pik", 'w') as f:
                #     pickle.dump((train_loss, train_acc, val_loss, val_acc), f)

                # Early Stopping Condition
                if best_val > 0.95:
                    break
        # Test Loop
        dic = data.get_dic_test(entity_net.S,entity_net.Q,entity_net.A)
        test_loss, test_acc = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)

        # Print and Write Test Loss/Accuracy
        logging.info("Test Loss: %.3f\tTest Accuracy: %.3f" % (test_loss, test_acc))
        return train_loss, train_acc, val_loss, val_acc,test_loss, test_acc



def main():
    embedding_size = 50
    sent_len = 10
    sent_numb = 20
    batch_size = 32
    epoch = 200
    data = Dataset(train_size=1000,dev_size=10,test_size=10,sent_len=sent_len,sent_numb=sent_numb, embedding_size= embedding_size)

    par = dict(
    vocab_size = data._data["vocab_size"],
    label_num = data._data["label_num"],
    num_blocks = 20,
    sent_len = sent_len,
    sent_numb = sent_numb,
    embedding_size = embedding_size,
    embeddings_mat = data._data["embeddings_mat"],
    learning_rate=  0.001,
    clip_gradients= 40.0,
    opt = 'Adam', #RMSProp,
    trainable = [1,1,0,0],
    max_norm = None,
    no_out = False,
    decay_steps = 0, #  [epoch* data._data['len_training']/25,epoch* data._data['len_training']/100],,
    decay_rate = 0.5
    )


    train(epoch,batch_size, data,par)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main()


    # logging.info('-' * 50)
    # logging.info('Load data files..')
    #
    # logging.info('*' * 10 + ' Train')
    # train_examples = load_data('data/train.txt', 2, relabeling=True)
    #
    # logging.info('*' * 10 + ' Dev')
    # dev_examples = load_data('data/dev.txt', 1, relabeling=True)
    #
    #
    # num_train = len(train_examples[0])
    # num_dev = len(dev_examples[0])
    #
    #
    # logging.info('-' * 50)
    # logging.info('Build dictionary..')
    # word_dict = build_dict(train_examples[0] + train_examples[1])
    # entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')] + train_examples[2]))
    # entity_markers = ['<unk_entity>'] + entity_markers
    # entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    # logging.info('Entity markers: %d' % len(entity_dict))
    # num_labels = len(entity_dict)
    #
    # logging.info('-' * 50)
    # # Load embedding file
    # embeddings = gen_embeddings(word_dict, embedding_size, embedding_file)
    # (vocab_size, embedding_size) = embeddings.shape
    # par['vocab_size'] = vocab_size
    # par['embeddings_mat'] = embeddings
    # par['label_num'] = num_labels
    #
    # # vectorize Data
    # logging.info('-' * 50)
    # logging.info('Vectorize training..')
    # train_x1, train_x2, train_l, train_y = vectorize(train_examples, word_dict, entity_dict, sent_len, sent_numb)
    #
