from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time
import datetime
from src.utils.utils import Dataset, gen_embeddings
from src.EN import EntityNetwork
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import cPickle as pickle
import logging
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
from sklearn.grid_search import ParameterGrid
import cPickle as pickle
import logging
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
sns.set_style(style='white')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(epoch,batch_size, data,par,dr,_test):

    tf.reset_default_graph()
    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)

        logging.info('Initializing Variables!')
        sess.run(tf.global_variables_initializer())
        all_train = data.gen_examples(batch_size,'train')

        logging.info('Training stated')
        for idx, (mb_x1, mb_x2, mb_y) in enumerate(all_train):
            s_s=[]
            for m in mb_x1[0]:
                t= []
                for e in m:
                    if(e!=0):
                        t.append(data._data['vocab'][e-1])
                if(len(t)>0):
                    s_s.append(t)
            s_s= [" ".join(sss) for sss in s_s]

            q_q =[]
            for e2 in mb_x2[0][0]:
                if(e2!=0):
                    q_q.append(data._data['vocab'][e2-1])
            q_q = " ".join(q_q)
            print (q_q)
            dic = {entity_net.S:mb_x1,entity_net.Q:mb_x2,
                   entity_net.A:mb_y, entity_net.keep_prob:dr}
            k,o,s,q,l= sess.run([entity_net.keys,entity_net.out,entity_net.story_embeddings,entity_net.query_embedding,entity_net.length],feed_dict=dic)
            gs=[]
            for i in range(l):
                temp = np.split(o[0][i], len(k))
                g =[]
                for j in range(len(k)):
                    g.append(sigmoid(np.inner(s[0][i],temp[j])+np.inner(s[0][i],k[j])+np.inner(s[0][i],q[0][0])))
                gs.append(g)

            plt.figure(figsize=(15,7.5))
            ax = sns.heatmap(np.transpose(np.array(gs)),cmap="YlGnBu",vmin=0, vmax=1)
            ax.set_xticks( [ i for i in range(len(s_s)) ] )
            ax.set_xticklabels( s_s, rotation=45 )
            ax.set_yticklabels([ i+1 for i in range(len(k)) ],rotation=0 )

            plt.title(q_q+"?")
            plt.tight_layout()
            plt.show()




def get_parameters(data,epoch,sent_len,sent_numb,embedding_size, params):
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """
    embedding_file = 'data/glove.6B.{}d.txt'.format(embedding_size)
    # embeddings_mat = gen_embeddings(data._data['word_idx'], embedding_size, embedding_file)
    dists = dict(
    vocab_size = data._data["vocab_size"],label_num = data._data["vocab_size"],
    sent_len = sent_len, sent_numb = sent_numb, embedding_size = embedding_size,
    embeddings_mat = [], clip_gradients= 40.0,
    max_norm = None, no_out = False, decay_steps = 0, decay_rate = 0, opt = 'Adam',
    num_blocks = params['nb'],
    learning_rate= params['lr'],
    trainable = params['tr'],
    L2 = params['L2']
    )
    return dists

def main(task_num,sample_size=''):
    embedding_size = 100
    epoch = 1
    best_accuracy = 0.0
    grind_ris={}

    if not os.path.exists('data/ris/task_{}'.format(task_num)):
        os.makedirs('data/ris/task_{}'.format(task_num))

    param_grid = {'nb': [2],
                  'lr': [0.001],
                  'tr': [[0,0,0,0]],
                  'L2': [0.0],# [0.0,0.1,0.01,0.001,0.0001]
                  'bz': [1],
                  'dr': [0.5],
                  }

    grid = list(ParameterGrid(param_grid))
    np.random.shuffle(grid)
    for params in list(grid):
        data = Dataset('data/tasks_1-20_v1-2/en-valid{}/'.format(sample_size),int(task_num))

        ## for sentence
        par = get_parameters(data,epoch,data._data['sent_len'],data._data['sent_numb'],embedding_size,params)
        train(epoch,params['bz'], data, par, dr=params['dr'], _test=True)




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    main(task_num=sys.argv[1])
