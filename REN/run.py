from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import datetime
from src.utils import Dataset,get_emb_matrix
from src.EN import EntityNetwork
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import cPickle as pickle

def train(epoch,batch_size, data,par):
    tf.reset_default_graph()
    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)
        print('Init stated')
        sess.run(tf.global_variables_initializer())
        train_loss, train_acc, val_loss, val_acc = {}, {}, {}, {}

        print('Training stated')
        best_val=0.0
        for e in range(epoch):
            loss, acc, counter = 0.0, 0.0, 0
            for i, elem in enumerate(data.get_batch_train(batch_size)):
                dic = data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,elem[0],elem[1])
                curr_loss, curr_acc, _ = sess.run([entity_net.loss_val, entity_net.accuracy, entity_net.train_op],
                                                  feed_dict=dic)
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 10 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f" % (e, counter, loss / float(counter), acc / float(counter)))
            # Add train loss, train acc to data
            train_loss[e], train_acc[e] = loss / float(counter), acc / float(counter)

            # Increment Epoch
            sess.run(entity_net.epoch_increment)


            if e % 10 == 0:
                dic = data.get_dic_val(entity_net.S,entity_net.Q,entity_net.A)
                val_loss_val, val_acc_val = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
                print ("Epoch %d Validation Loss: %.3f\tValidation Accuracy: %.3f" % (e, val_loss_val, val_acc_val))

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
        print ("Test Loss: %.3f\tTest Accuracy: %.3f" % (test_loss, test_acc))
        return train_loss, train_acc, val_loss, val_acc,test_loss, test_acc


def get_random_parameters(data,epoch):
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """

    # change this dictionary to change the random distribution
    dists = dict(
        vocab_size = [data._data["vocab_size"]],
        sent_len = [data._data['sent_len']],
        sent_numb = [data._data['sent_numb']],
        num_blocks = [data._data["vocab_size"],20,],
        embedding_size =  [50,100],
        learning_rate=  [0.01,0.001,0.0001],
        clip_gradients= [40.0],
        opt=  ['Adam'], #RMSProp
        trainable=  [[0,0,0,0],[1,0,1,0],[1,1,1,1],[1,0,1,1],[1,0,0,0]],
        max_norm=  [None,1],
        no_out=  [True,False],
        decay_steps=  [epoch* data._data['len_training']/25,epoch* data._data['len_training']/100],
        decay_rate=  [0.1,0.5],
    )

    d = {}
    for k in dists:
        if hasattr(dists[k], '__len__'):
            idx = random.randint(0, len(dists[k]) - 1)
            d[k] = dists[k][idx]
        else:
            d[k] = dists[k].rvs()
    if d['no_out']:
        d['num_blocks'] = data._data["vocab_size"]
    return d


def main(task_num,sample_size=''):
    batch_size = 32
    epoch = 200

    task_num = 1
    if not os.path.exists('data/ris/task_{}'.format(task_num)):
        os.makedirs('data/ris/task_{}'.format(task_num))
    data = Dataset('data/tasks_1-20_v1-2/en-valid{}/'.format(sample_size),task_num)


    for i in range(10):
        par = get_random_parameters(data,epoch)
        print(par)
        if (par['trainable'][0]==1):
            embeddings_mat = get_emb_matrix(data._data["vocab_size"],data._data["word_idx"],par['embedding_size'],emb_file='data/glove.6B.{}d.txt'.format(par['embedding_size']))
            par['embeddings_mat'] =embeddings_mat

        train_loss, train_acc, val_loss, val_acc,test_loss, test_acc = train(epoch,batch_size, data, par)
        par['id'] = 'Par'+ str(i)
        par['acc'] = sorted([[k,v] for k,v in val_acc.items()])[-1][1]
        pickle.dump( [train_loss, train_acc, val_loss, val_acc,test_loss, test_acc,par], open( 'data/ris/task_{}/{}.p'.format(task_num,par['id']), "wb" ) )



if __name__ == '__main__':
    main(task_num=sys.argv[1])
