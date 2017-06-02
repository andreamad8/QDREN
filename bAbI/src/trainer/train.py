from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time
import datetime
from src.EN import EntityNetwork
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import cPickle as pickle
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
sns.set_style(style='white')


def train(epoch,batch_size, data,par,dr, _test):

    def val_test(d,ty):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def viz(epoch):
            s_s=[]
            ### mb_x1 input senteces
            for m in mb_x1[0]:
                t = []
                for e in m:
                    if(e != 0):
                        t.append(data._data['vocab'][e-1])
                if(len(t) > 0):
                    s_s.append(t)

            s_s = [" ".join(sss) for sss in s_s]

            q_q = []
            for e2 in mb_x2[0][0]:
                if(e2 != 0):
                    q_q.append(data._data['vocab'][e2-1])
            q_q = " ".join(q_q)

            k,o,s,q,l = sess.run([entity_net.keys,entity_net.out,entity_net.story_embeddings,entity_net.query_embedding,entity_net.length],feed_dict=dic)
            gs=[]

            for i in range(l):
                temp = np.split(o[0][i], len(k))
                g =[]
                for j in range(len(k)):
                    g.append(sigmoid(np.inner(s[0][i],temp[j])+np.inner(s[0][i],k[j])+np.inner(s[0][i],q[0][0])))
                gs.append(g)

            plt.figure(figsize=(15,7.5))
            ax = sns.heatmap(np.transpose(np.array(gs)),cmap="YlGnBu",vmin=0, vmax=1)
            ax.set_xticks([i for i in range(len(s_s))])
            ax.set_xticklabels(s_s,rotation=45)
            ax.set_yticklabels([ i+1 for i in range(len(k)) ],rotation=0 )

            plt.title(q_q+"?",fontsize=20)
            plt.tight_layout()
            plt.savefig('data/plot/ep%d.png'%int(epoch), format='png', dpi=300)
            plt.close()

        _loss_val, _acc_val, _counter = 0.0, 0.0, 0
        for idx, (mb_x1, mb_x2, mb_y) in enumerate(d):
            dic = {entity_net.S:mb_x1,entity_net.Q:mb_x2,
                   entity_net.A:mb_y, entity_net.keep_prob:1.0}
            curr_loss_val, curr_acc_val = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
            _loss_val, _acc_val, _counter = _loss_val + curr_loss_val, _acc_val + curr_acc_val, _counter + 1
        if(ty=='Validation' and e % 1 ==0 ):
            logging.info("Start plotting")
            viz(e)
        logging.info("Epoch %d\t%s Loss: %.3f\t %s Accuracy: %.3f" % (e,ty[:4],_loss_val / float(_counter),ty, _acc_val/float(_counter)))
        return _loss_val / float(_counter), _acc_val/float(_counter)


    def tr(verbose):
        loss, acc, counter = 0.0, 0.0, 0
        np.random.shuffle(all_train)
        for idx, (mb_x1, mb_x2, mb_y) in enumerate(all_train):
            dic = {entity_net.S:mb_x1,entity_net.Q:mb_x2,
                   entity_net.A:mb_y, entity_net.keep_prob:dr}
            curr_loss, curr_acc, _ = sess.run([entity_net.loss_val, entity_net.accuracy, entity_net.train_op],
                                              feed_dict=dic)
            loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
            if counter % verbose == 0:
                logging.info("Epoch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f" % (e, loss / float(counter), acc / float(counter)))
        return loss / float(counter), acc / float(counter)

    def init():
        # Setup Checkpoint + Log Paths

        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = {}, {}, {}, {}, {}, {}
        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

    tf.reset_default_graph()
    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = init()


        logging.info('Initializing Variables!')
        sess.run(tf.global_variables_initializer())

        # Get Current Epoch
        curr_epoch = sess.run(entity_net.epoch_step)

        logging.info('Training stated')
        all_train = data.gen_examples(batch_size,'train')
        all_val   = data.gen_examples(1,'val')
        all_test  = data.gen_examples(batch_size,'test')
        best_val,patient= 0.0, 0.0
        for e in range(curr_epoch,epoch):
            train_loss[e], train_acc[e] = tr(20)
            val_loss[e], val_acc[e] = val_test(all_val,'Validation')
            if (_test):
                test_loss[e], test_acc[e] = val_test(all_test,'Test')
            # Update best_val
            if val_acc[e] > best_val:
                best_val, patient = val_acc[e], 0
            elif val_acc[e] > best_val:
                patient += 0.5
            else:
                patient += 1.0

            # Early Stopping Condition
            if patient > 100.0:
                break
            sess.run(entity_net.epoch_increment)

        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


# for i, elem in enumerate(data.get_batch_train(batch_size,'train')):
# data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,entity_net.keep_prob,elem[0],elem[1],dr)
