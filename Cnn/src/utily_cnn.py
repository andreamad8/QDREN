from __future__ import absolute_import
from itertools import chain
from six.moves import range, reduce
from gensim.models import Word2Vec
import gensim
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
import codecs
import os
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cPickle as pickle

class Dataset():
    def __init__(self, data='data/cnn/questions/'):
        self.emb = self.get_vocab_and_index(emb_file='data/emb_100_CBOW_wind_5.txt')
        self._data = self.get_train_test(data,max_sent=20,n_entities=550)
        self.len_train = len(self._data['train']['S'])
        self.len_val = len(self._data['val']['S'])
        self.len_test = len(self._data['test']['S'])
        self.num_batches=0



    def make_batches(self,size, batch_size):
        """Returns a list of batch indices (tuples of indices).
        # Arguments
            size: Integer, total size of the data to slice into batches.
            batch_size: Integer, batch size.
        # Returns
            A list of tuples of array indices.
        """
        self.num_batches = int(np.ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, self.num_batches)]

    def unison_shuffled_copies(self,a, b,c):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p]

    def get_batch_train(self,batch_size):
        #randomize = np.arange(self.len_train)
        #np.random.shuffle(randomize)
        #self._data['train']['S'] = self._data['train']['S'][randomize]
        #self._data['train']['Q'] = self._data['train']['Q'][randomize]
        #self._data['train']['A'] = self._data['train']['A'][randomize]
        return self.make_batches(self.len_train, batch_size)

    def get_dic_train(self,S_input,Q_input,A_input,k,t):
        # S,Q,A = self.get_data('data/cnn/questions/training',
        #                     self._data['train'][k:t],
        #                     max_sent=20,
        #                     embedding_size=100,
        #                     n_entities=550)
        # return {S_input:S,
        #         Q_input:Q,
        #         A_input:A}

        return {S_input:self._data['train']['S'][k:t],
                Q_input:self._data['train']['Q'][k:t],
                A_input:self._data['train']['A'][k:t]}

    def get_dic_val(self,S_input,Q_input,A_input):
        return {S_input:self._data['val']['S'],
                Q_input:self._data['val']['Q'],
                A_input:self._data['val']['A']}

    def get_dic_test(self,S_input,Q_input,A_input):
        return {S_input:self._data['test']['S'],
                Q_input:self._data['test']['Q'],
                A_input:self._data['test']['A']}




    def get_train_test(self,which_task='data/cnn/questions/',max_sent=20,n_entities=550):


        # S_train,Q_train,A_train = self.get_data(which_task+'training',
        #                                          os.listdir('data/cnn/questions/training'),
        #                                          max_sent=max_sent,
        #                                          embedding_size=100,
        #                                          n_entities=n_entities)
        data_train = pickle.load( open( "data/cnn/training_set.p", "rb" ) )
        S_train,Q_train,A_train = data_train[0],data_train[1],data_train[2]
        print("IMPORTING TRAINING DONE")

        S_validation,Q_validation,A_validation = self.get_data(which_task+'validation',
                                                 os.listdir('data/cnn/questions/validation'),
                                                 max_sent=max_sent,
                                                 embedding_size=100,
                                                 n_entities=n_entities)
        print("IMPORTING VALIDATION DONE")

        S_test,Q_test,A_test = self.get_data(which_task+'test',
                            os.listdir('data/cnn/questions/test'),
                            max_sent=max_sent,
                            embedding_size=100,
                            n_entities=n_entities)
        print("IMPORTING TEST DONE")

        return {'train':{'S':S_train, 'Q':Q_train, 'A':A_train},
                'val':{'S':S_validation, 'Q':Q_validation, 'A':A_validation},
                'test':{'S':S_test, 'Q':Q_test, 'A':A_test},
                'sent_numb':max_sent}



    def get_vocab_and_index(self,emb_file):
        f = codecs.open(emb_file, "r", "utf-8")
        # load matrix
        embeddings_index = {}
        vocab = []
        embeddings_index["<UNK>".decode('utf-8')]= np.zeros(100)
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vocab.append(word)
            coefs = np.asarray(values[1:]).astype(np.float)
            embeddings_index[word] = coefs
        f.close()

        return embeddings_index

    def get_data(self,dirname,file_list,max_sent=20,embedding_size=100,n_entities=550):
        S = []
        Q = []
        A = []
        for fname in file_list:
            data=[]
            filename=os.path.join(dirname, fname)
            for line in codecs.open(filename,'r', 'utf-8'):
    	           data.append(line[:-1])
            story_ent    = sent_tokenize(data[2])
            question_ent = sent_tokenize(data[4])
            story_ent = [s.lower().split() for s in story_ent]
            question_ent = [s.lower().split() for s in question_ent]
            ans_vec = np.zeros(n_entities)
            ans = data[6].replace(" ", "_")

            ent = {data[i].split(":")[0]: data[i].split(":")[1].lower()  for i in range(8,len(data))}
            question_ent= [question_ent[0]]
            for a in range(len(question_ent)):
                for b in range(len(question_ent[a])):
                    if question_ent[a][b] == "@placeholder":
                        question_ent[a][b] = self.emb[ "<UNK>".decode('utf-8')]
                    else:
                        question_ent[a][b] = self.emb[question_ent[a][b]]
                question_ent[a] = np.sum(np.asarray(question_ent[a]),0)


            k = 0
            ris_ent_sorted_by_postion = {v:[]for k,v in ent.items()}
            for j in range(len(story_ent)):
                for i in range(len(story_ent[j])):
                    if(story_ent[j][i] in ent):
                        ris_ent_sorted_by_postion[ent[story_ent[j][i]]].append(k)
                        if(ent[story_ent[j][i]]!=''):
                            story_ent[j][i] = self.emb[ent[story_ent[j][i]].replace(" ", "_")]
                        else:
                            story_ent[j][i] = np.zeros(embedding_size)
                        k += 1
                    else:
                        if(story_ent[j][i]!=''):
                            story_ent[j][i] = self.emb[story_ent[j][i]]
                        else:
                            story_ent[j][i] = np.zeros(embedding_size)

                story_ent[j] = np.sum(np.asarray(story_ent[j]),0)
            for i in range(len(story_ent),max_sent):
                story_ent.append(np.zeros(embedding_size))

            temp={}
            for k,v in ris_ent_sorted_by_postion.items():
                if(len(v)>0):
                    temp[k]=min(v)
            temp= sorted(temp, key=temp.get)
            for i in range(len(temp)):
                ris_ent_sorted_by_postion[temp[i]] = i
            index_anw = ris_ent_sorted_by_postion[ent[ans]]
            ans_vec[index_anw]=1

            # print(np.array(story_ent).shape)
            S.append(np.array(story_ent[:max_sent]))
            Q.append(np.array(question_ent))
            A.append(ans_vec)


        return np.array(S),np.array(Q),np.array(A)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        # mean = tf.reduce_mean(var)
        # tf.scalar_summary('mean/' + name, mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.scalar_summary('stddev/' + name, stddev)
        # tf.scalar_summary('max/' + name, tf.reduce_max(var))
        # tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
