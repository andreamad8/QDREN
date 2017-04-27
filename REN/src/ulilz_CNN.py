import numpy as np
import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
from iteration_utilities import flatten

class Dataset():
    def __init__(self,train_size,dev_size,test_size,sent_len,sent_numb,embedding_size):
        self._data = get_train_test(train_size,dev_size,test_size,sent_len,sent_numb,embedding_size)
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

    def get_batch_train(self,batch_size,data):
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

    def get_dic_val(self,S_input,Q_input,A_input,keep_prob,i,j):
        return {S_input:self._data['val']['S'][i:j],
                Q_input:self._data['val']['Q'][i:j],
                A_input:self._data['val']['A'][i:j],
                keep_prob:1.0}

    def get_dic_test(self,S_input,Q_input,A_input,keep_prob,i,j):
        return {S_input:self._data['test']['S'][i:j],
                Q_input:self._data['test']['Q'][i:j],
                A_input:self._data['test']['A'][i:j],
                keep_prob:1.0}


def get_train_test(train_size,dev_size,test_size,sent_len,sent_numb,embedding_size):
    embedding_file = 'data/glove.6B.{}d.txt'.format(embedding_size)
    logging.info('-' * 50)
    logging.info('Load data files..')

    logging.info('*' * 10 + ' Train')
    train_examples = load_data('data/train.txt', train_size, relabeling=True)

    logging.info('*' * 10 + ' Dev')
    dev_examples = load_data('data/dev.txt', dev_size, relabeling=True)

    logging.info('*' * 10 + ' Test')
    test_examples = load_data('data/test.txt', test_size, relabeling=True)


    num_train = len(train_examples[0])
    num_dev = len(dev_examples[0])
    num_dev = len(test_examples[0])



    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = build_dict(train_examples[0] + train_examples[1] + dev_examples[0] + dev_examples[1] + test_examples[0] + test_examples[1])
    entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')] + train_examples[2]))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    logging.info('Entity markers: %d' % len(entity_dict))
    num_labels = len(entity_dict)

    logging.info('-' * 50)
    # Load embedding file
    embeddings = gen_embeddings(word_dict, embedding_size, embedding_file)
    (vocab_size, embedding_size) = embeddings.shape

    # vectorize Data
    logging.info('-' * 50)
    logging.info('Vectorize training..')
    train_x1, train_x2, train_l, train_y = vectorize(train_examples, word_dict, entity_dict, sent_len, sent_numb)
    dev_x1, dev_x2, dev_l, dev_y = vectorize(dev_examples, word_dict, entity_dict, sent_len, sent_numb)
    test_x1, test_x2, test_l, test_y = vectorize(test_examples, word_dict, entity_dict, sent_len, sent_numb)

    return {'train':{'S':train_x1, 'Q':train_x2, 'A':train_y},
            'val':{'S':dev_x1, 'Q':dev_x2, 'A':dev_y},
            'test':{'S':test_x1, 'Q':test_x2, 'A':test_y},
            'vocab_size':vocab_size,
            'sent_len':sent_len,
            'sent_numb':sent_numb,
            'word_idx':word_dict,
            'len_training':num_train,
            'embeddings_mat': embeddings,
            'label_num':num_labels}




def load_data(in_file, max_example=None, relabeling=True):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    num_examples = 0
    f = open(in_file, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip()
        document = f.readline().strip().lower()

        if relabeling:
            q_words = question.split(' ')
            d_words = document.split(' ')
            assert answer in d_words

            entity_dict = {}
            entity_id = 0
            for word in d_words + q_words:
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
            d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
            answer = entity_dict[answer]

            question = ' '.join(q_words)
            document = ' '.join(d_words)

        if (entity_id <=50):
            questions.append(question)
            answers.append(answer)
            documents.append(document)
            num_examples += 1

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, answers)


def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def vectorize(examples, word_dict, entity_dict, max_s_len, max_s_numb,
                sort_by_len=True, verbose=True):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(examples[0]), len(entity_dict)))
    in_y = []

    # stat_len =[]
    # stat_wordxsent = []
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        d_sents = d.split(' . ')
        for i,s in enumerate(d_sents):
            d_sents[i]= s.split(' ')
        # stat_len.append(len(d_sents))
        # stat_wordxsent.append(max([len(s)for s in d_sents]))
        # d_words = d.split(' ')
        q_words = q.split(' ')
        assert (a in flatten(d_sents))

        for i,s in enumerate(d_sents):
            ls = max(0, max_s_len - len( s ) )
            d_sents[i] = [word_dict[w] if w in word_dict else 0 for w in s] + [0] * ls
            d_sents[i] = d_sents[i][:max_s_len]

        # pad to memory_size
        lm = max(0, max_s_numb - len(d_sents))
        for _ in range(lm):
            d_sents.append([0] * max_s_len)
        d_sents = d_sents[:max_s_numb]
        # seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
        # seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]

        ls = max(0, max_s_len - len( q_words ) )
        q_words = [word_dict[w] if w in word_dict else 0 for w in q_words] + [0] * ls
        q_words = q_words[:max_s_len]


        if (len(d_sents) > 0) and (len(q_words) > 0):
            in_x1.append(d_sents)
            in_x2.append(q_words)
            in_l[idx, [entity_dict[w] for w in flatten(d_sents) if w in entity_dict]] = 1.0
            in_y.append(entity_dict[a] if a in entity_dict else 0)
        if verbose and (idx % 100000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[0])))
    # logging.info('Max sent:{}\t Avg sent: {} Std sent:{}'.format(max(stat_len),sum(stat_len)/len(stat_len),np.std(stat_len)))
    # logging.info('Max wxse:{}\t Avg wxse: {} Std wxse:{}'.format(max(stat_wordxsent),sum(stat_wordxsent)/len(stat_wordxsent),np.std(stat_wordxsent)))

    # def len_argsort(seq):
    #     return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    #
    # if sort_by_len:
    #     # sort by the document length
    #     sorted_index = len_argsort(in_x1)
    #     in_x1 = [in_x1[i] for i in sorted_index]
    #     in_x2 = [in_x2[i] for i in sorted_index]
    #     in_l = in_l[sorted_index]
    #     in_y = [in_y[i] for i in sorted_index]

    return np.array(in_x1), np.expand_dims(np.array(in_x2), axis=1), in_l, np.array(in_y)


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(config._floatX)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def gen_embeddings(word_dict, dim, in_file=None,
                   init=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
    embeddings = np.zeros((num_words, dim))
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


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic
