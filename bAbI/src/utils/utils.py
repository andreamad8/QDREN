from __future__ import absolute_import
from itertools import chain
from six.moves import range, reduce
import os
import re
import numpy as np
import tensorflow as tf
import codecs
import sys
from tqdm import tqdm
import pickle
import logging
class Dataset():
    def __init__(self, data='data/tasks_1-20_v1-2/en/',ts_num=1):
        self._data = get_train_test(data,ts_num)
        self.len_train = len(self._data['train']['S'])
        self.len_val = len(self._data['val']['S'])
        self.len_test = len(self._data['test']['S'])



    def get_minibatches(self,n, minibatch_size, shuffle=True):
        idx_list = np.arange(0, n, minibatch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
        return minibatches

    def gen_examples(self,batch_size,tip):
        """
            Divide examples into batches of size `batch_size`.
        """
        minibatches = self.get_minibatches(len(self._data[tip]['S']), batch_size)
        all_ex = []
        for minibatch in minibatches:
            mb_x1 = [self._data[tip]['S'][t] for t in minibatch]
            mb_x2 = [self._data[tip]['Q'][t] for t in minibatch]
            mb_y =  [self._data[tip]['A'][t] for t in minibatch]
            all_ex.append((np.array(mb_x1), np.array(mb_x2), np.array(mb_y)))
        return all_ex


def get_train_test(which_task='data/tasks_1-20_v1-2/en/',task_num=1):
    train, val, test = load_task(which_task,task_num)
    data = train + test + val

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    if (task_num==3):
        max_story_size = min(130, max_story_size)
    else:
        max_story_size = min(70, max_story_size)


    vocab_size = len(word_idx) +1# +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position
    sentence_size+=1
    logging.info("Longest sentence length: "+ str( sentence_size))
    logging.info("Longest story length: "+ str( max_story_size))
    logging.info("Average story length: "+ str( mean_story_size))
    logging.info("Training sample: "+ str(len(train)))
    logging.info("Validation sample: "+ str(len(val)))
    logging.info("Test sample: "+ str(len(test)))
    logging.info("Vocab size : "+ str(vocab_size))


    S, Q, A = vectorize_data(train, word_idx, sentence_size, max_story_size)
    valS, valQ, valA = vectorize_data(val, word_idx, sentence_size, max_story_size)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, max_story_size)
    return {'train':{'S':S, 'Q':np.expand_dims(Q, axis=1), 'A':A},
            'val':{'S':valS, 'Q':np.expand_dims(valQ, axis=1), 'A':valA},
            'test':{'S':testS, 'Q':np.expand_dims(testQ, axis=1), 'A':testA},
            'vocab':vocab,
            'vocab_size':vocab_size,
            'sent_len':sentence_size,
            'sent_numb':max_story_size,
            'word_idx':word_idx,
            'len_training':len(train)}


def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]

    s = 'qa{}_'.format(task_id)

    train_file = [f for f in files if s in f and 'train' in f][0]
    val_file = [f for f in files if s in f and 'valid.txt' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    val_data = get_stories(val_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, val_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which
        # corresponds to vector of lookup table
        #for i in range(len(ss)):
        #    ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        # y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        # print(answer)
        # for a in answer:
        #     y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(word_idx[answer[0]])
    return np.array(S), np.array(Q), np.array(A)


def gen_embeddings(word_dict, dim, in_file=None, init=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) +1 
    # embeddings = np.zeros((num_words, dim))
    embeddings = np.random.standard_normal(size=(num_words, dim))
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings

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
        tf.summary.histogram(name, var)
