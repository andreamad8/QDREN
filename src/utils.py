from __future__ import absolute_import
from itertools import chain
from six.moves import range, reduce
import os
import re
import numpy as np
import tensorflow as tf

class Dataset():
    def __init__(self, data='data/tasks_1-20_v1-2/en/'):
        self._data = get_train_test(data)
        self.len_train = len(self._data['train']['S'])

    def make_batches(self,size, batch_size):
        """Returns a list of batch indices (tuples of indices).
        # Arguments
            size: Integer, total size of the data to slice into batches.
            batch_size: Integer, batch size.
        # Returns
            A list of tuples of array indices.
        """
        num_batches = int(np.ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, num_batches)]

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

    def get_dic_train(self,S_input,Q_input,A_input,i,j):
        return {S_input:self._data['train']['S'][i:j],
                Q_input:self._data['train']['Q'][i:j],
                A_input:self._data['train']['A'][i:j]}

    def get_dic_val(self,S_input,Q_input,A_input):
        return {S_input:self._data['val']['S'],
                Q_input:self._data['val']['Q'],
                A_input:self._data['val']['A']}

    def get_dic_test(self,S_input,Q_input,A_input):
        return {S_input:self._data['test']['S'],
                Q_input:self._data['test']['Q'],
                A_input:self._data['test']['A']}


def get_train_test(which_task='data/tasks_1-20_v1-2/en/'):
    train, val, test = load_task(which_task,1)
    data = train + test + val

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    max_story_size = min(50, max_story_size)


    vocab_size = len(word_idx) +1# +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position
    sentence_size+=1
    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)
    print("Training sample",len(train))
    print("Validation sample",len(val))
    print("Test sample",len(test))



    # train/validation/test sets
    S, Q, A = vectorize_data(train, word_idx, sentence_size, max_story_size)
    valS, valQ, valA = vectorize_data(val, word_idx, sentence_size, max_story_size)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, max_story_size)
    return {'train':{'S':S, 'Q':np.expand_dims(Q, axis=1), 'A':A},
            'val':{'S':valS, 'Q':np.expand_dims(valQ, axis=1), 'A':valA},
            'test':{'S':testS, 'Q':np.expand_dims(testQ, axis=1), 'A':testA},
            'vocab':vocab,
            'vocab_size':vocab_size,
            'sent_len':sentence_size,
            'sent_numb':max_story_size}

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
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
