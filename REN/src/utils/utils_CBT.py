import numpy as np
import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
import re
from iteration_utilities import flatten
# import seaborn as sns
# import matplotlib.pyplot as plt

class Dataset():
    def __init__(self,train_size,dev_size,test_size,sent_len,sent_numb,embedding_size,max_windows,win,ty_CN_NE):
        self._data = get_train_test(train_size,dev_size,test_size,sent_len,sent_numb,embedding_size,max_windows=max_windows, win=win,ty_CN_NE=ty_CN_NE)
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

    def get_batch(self,batch_size,data):
        randomize = np.arange(len(self._data[data]['S']))
        np.random.shuffle(randomize)
        self._data[data]['S'] = self._data[data]['S'][randomize]
        self._data[data]['Q'] = self._data[data]['Q'][randomize]
        self._data[data]['A'] = self._data[data]['A'][randomize]
        return self.make_batches(len(self._data[data]['S']), batch_size)

    def get_dic_train(self,S_input,Q_input,A_input,keep_prob,i,j,dr):
        return {S_input:self._data['train']['S'][i:j],
                Q_input:self._data['train']['Q'][i:j],
                A_input:self._data['train']['A'][i:j],
                keep_prob:dr}

    def get_dic_val_test(self,S_input,Q_input,A_input,keep_prob,i,j,ty):
        return {S_input:self._data[ty]['S'][i:j],
                Q_input:self._data[ty]['Q'][i:j],
                A_input:self._data[ty]['A'][i:j],
                keep_prob:1.0}

    # def get_dic_test(self,S_input,Q_input,A_input,keep_prob,i,j):
    #     return {S_input:self._data['test']['S'][i:j],
    #             Q_input:self._data['test']['Q'][i:j],
    #             A_input:self._data['test']['A'][i:j],
    #             keep_prob:1.0}
    #

    # def get_minibatches(self,n, minibatch_size, shuffle=True):
    #     idx_list = np.arange(0, n, minibatch_size)
    #     if shuffle:
    #         np.random.shuffle(idx_list)
    #     minibatches = []
    #     for idx in idx_list:
    #         minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    #     return minibatches
    #
    # def gen_examples(self,batch_size,tip):
    #     """
    #         Divide examples into batches of size `batch_size`.
    #     """
    #     minibatches = self.get_minibatches(len(self._data[tip]['S']), batch_size)
    #     all_ex = []
    #     for minibatch in minibatches:
    #         mb_x1 = [self._data[tip]['S'][t] for t in minibatch]
    #         mb_x2 = [self._data[tip]['Q'][t] for t in minibatch]
    #         mb_y =  [self._data[tip]['A'][t] for t in minibatch]
    #         mb_c =  [self._data[tip]['C'][t] for t in minibatch]
    #
    #         all_ex.append((np.array(mb_x1), np.array(mb_x2), np.array(mb_y)))
    #     return all_ex





def get_train_test(train_size,dev_size,test_size,sent_len,sent_numb,
                   embedding_size,max_windows, win, ty_CN_NE):

    logging.info('-' * 50)
    logging.info('Load data files..')
    logging.info('*' * 10 + ' Train')
    train_examples = load_data('data/cbtest_{}_train.txt'.format(ty_CN_NE), train_size)

    logging.info('*' * 10 + ' Dev')
    dev_examples = load_data('data/cbtest_{}_valid_2000ex.txt'.format(ty_CN_NE), dev_size)

    logging.info('*' * 10 + ' Test')
    test_examples = load_data('data/cbtest_{}_test_2500ex.txt'.format(ty_CN_NE), test_size)


    num_train = len(train_examples[0])
    num_dev = len(dev_examples[0])
    num_dev = len(test_examples[0])


    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = build_dict(train_examples[0] + [train_examples[1]] + dev_examples[0] + [dev_examples[1]] + test_examples[0] + [test_examples[1]],win=win)

    logging.info('-' * 50)
    if(embedding_size not in [50,100,200,300]):
        vocab_size = max(word_dict.values()) + 1
        embeddings = []
    else:
        # Load embedding file
        embedding_file = 'data/glove.6B.{}d.txt'.format(embedding_size)
        embeddings = gen_embeddings(word_dict, embedding_size, embedding_file)
        (vocab_size, embedding_size) = embeddings.shape

    # vectorize Data
    logging.info('-' * 50)
    logging.info('Vectorize training..')
    if (win != None):
        train_x1, train_x2, train_y, train_c = vectorize_window(train_examples, word_dict, max_windows=max_windows, win=win)
        dev_x1, dev_x2, dev_y, dev_c = vectorize_window(dev_examples, word_dict, max_windows=max_windows, win=win)
        test_x1, test_x2, test_y, test_c = vectorize_window(test_examples, word_dict, max_windows=max_windows, win=win)

    else:

        train_x1, train_x2, train_y, train_c = vectorize(train_examples, word_dict, sent_len, sent_numb)
        dev_x1, dev_x2, dev_y, dev_c = vectorize(dev_examples, word_dict, sent_len, sent_numb)
        test_x1, test_x2, test_y, test_c = vectorize(test_examples, word_dict, sent_len, sent_numb)

    # sns.distplot(train_y,bins=10)
    # sns.distplot(dev_y,bins=10)
    # sns.distplot(test_y,bins=10)
    #
    # plt.show()
    return {'train':{'S':train_x1, 'Q':train_x2, 'A':train_y, 'C': train_c},
            'val':  {'S':dev_x1,   'Q':dev_x2,   'A':dev_y  , 'C': dev_c},
            'test': {'S':test_x1,  'Q':test_x2,  'A':test_y , 'C': test_c},
            'vocab_size':vocab_size,
            'sent_len':sent_len,
            'sent_numb':sent_numb,
            'word_idx':word_dict,
            'len_training':num_train,
            'embeddings_mat': embeddings,
            'label_num':10}


def load_data(in_file, max_example=None):
    documents = []
    questions = []
    answer = []
    candidates = []
    num_examples = 0
    f = open(in_file, 'r')
    while(True):
        story = []
        for i in range(20):
            line = f.readline()
            if not line:
                break
            line = str.lower(line)
            nid, line = line.split(' ', 1)
            sent = line.rstrip().split(' ')
            story.append(sent)
        line = f.readline()
        if not line:
            break
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        line = line.split('\t')
        q = line[0].rstrip().split(' ')

        documents.append(story)
        questions.append(q)
        answer.append(line[1])
        candidates.append(line[3][:-1].split('|'))

        num_examples += 1
        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, answer, candidates)


def build_dict(sentences,win=None, max_words=60000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sents in sentences:
        for s in sents:
            for w in s:
                word_count[w] += 1

    ls = word_count.most_common(max_words)
    if (win != None):
        for j in range(win):
            ls.insert(0,('<begin-' + str(j) +'>',0))
        for j in range(win):
            ls.insert(0,('<end-' + str(j) +'>',0))

    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 1 for (index, w) in enumerate(ls)}

# def tokenize(sent):
#     '''Return the tokens of a sentence including punctuation.
#     >>> tokenize('Bob dropped the apple. Where is the apple?')
#     ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
#     '''
#     return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]
#

def gen_embeddings(word_dict, dim, in_file=None,
                   init=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
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


def vectorize(examples, word_dict, max_s_len, max_s_numb):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_y = []
    in_c = []

    for idx, (d, q, a, c) in enumerate(zip(examples[0], examples[1], examples[2], examples[3])):

        for i,s in enumerate(d):
            ls = max(0, max_s_len - len( s ) )
            d[i] = [word_dict[w] if w in word_dict else 0 for w in s] + [0] * ls
            d[i] = d[i][:max_s_len]

        ls = max(0, max_s_len - len( q ) )
        q = [word_dict[w] if w in word_dict else 0 for w in q] + [0] * ls
        q = q[:max_s_len]
        if (len(d) > 0) and (len(q) > 0):
            in_x1.append(d)
            in_x2.append(q)
            in_y.append(c.index(a))
            in_c.append([word_dict[w] if w in word_dict else 0 for w in c])

    return np.array(in_x1), np.expand_dims(np.array(in_x2), axis=1), np.array(in_y), np.array(in_c)


def vectorize_window(examples, word_dict, max_windows=10, win=2):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_y = []
    in_c = []
    # stat_len =[]
    for idx, (d, q, a, c) in enumerate(zip(examples[0], examples[1], examples[2], examples[3])):
        d_windows = []
        d = list(flatten(d))
        for j in range(win):
            d.insert(0,'<begin-' + str(j) +'>')
            d.append('<end-' + str(j) +'>')
            q.insert(0,'<begin-' + str(j) +'>')
            q.append('<end-' + str(j) +'>')

        for i in range(win,len(d)-win):
            if(d[i] in c):
                d_windows.append([word_dict[w] if w in word_dict else 0 for w in d[i-win:i+win+1]])

        # stat_len.append(len(d_windows))
        # pad to max_windows
        lm = max(0, max_windows - len(d_windows))
        for _ in range(lm):
            d_windows.append([0] * ((win*2)+1))
        d_windows = d_windows[:max_windows]

        for i in range(win,len(q)-win):
            if(q[i] == 'xxxxx'):
                q_windows = [word_dict[w] if w in word_dict else 0 for w in q[i-win:i+win+1]]

        if (len(d_windows) > 0) and (len(q_windows) > 0):
            in_x1.append(d_windows)
            in_x2.append(q_windows)
            in_y.append(c.index(a))
            in_c.append([word_dict[w] if w in word_dict else 0 for w in c])

    # logging.info('Max sent:{}\t Avg sent: {} Std sent:{}'.format(max(stat_len),sum(stat_len)/len(stat_len),np.std(stat_len)))
    return np.array(in_x1), np.expand_dims(np.array(in_x2), axis=1), np.array(in_y), np.array(in_c)
