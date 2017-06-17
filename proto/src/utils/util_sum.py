import numpy as np
import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
from iteration_utilities import flatten


class Dataset():
    def __init__(self,train_size,dev_size,test_size,sent_len,sent_numb,embedding_size,max_windows,win):
        self._data = get_train_test(train_size,dev_size,test_size,sent_len,sent_numb,embedding_size,max_windows=max_windows, win=win)

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

def get_train_test(train_size,dev_size,test_size,sent_len,sent_numb,embedding_size,max_windows, win):
    embedding_file = 'data/glove.6B.{}d.txt'.format(embedding_size)
    logging.info('-' * 50)
    logging.info('Load data files..')

    logging.info('*' * 10 + ' Train')
    train_examples = load_data('data/train.txt', train_size)

    logging.info('*' * 10 + ' Dev')
    dev_examples = load_data('data/dev.txt', dev_size)

    logging.info('*' * 10 + ' Test')
    test_examples = load_data('data/test.txt', test_size)


    num_train = len(train_examples[0])
    num_dev = len(dev_examples[0])
    num_dev = len(test_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = build_dict(train_examples[0] + train_examples[1] + dev_examples[0] + dev_examples[1] + test_examples[0] + test_examples[1],win=win)
    entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')] + train_examples[2]))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    logging.info('Entity markers: %d' % len(entity_dict))
    num_labels = len(entity_dict)

    logging.info('-' * 50)
    if(embedding_size not in [50,100,200,300]):
        vocab_size = max(word_dict.values()) + 1
        embeddings = []
    else:
        # Load embedding file
        embeddings = gen_embeddings(word_dict, embedding_size, embedding_file)
        (vocab_size, embedding_size) = embeddings.shape

    # vectorize Data
    logging.info('-' * 50)
    logging.info('Vectorize training..')
    if (win != None):
        train_x1, train_x2, train_l, train_y = vectorize_window(train_examples, word_dict, entity_dict, max_windows=max_windows, win=win)
        dev_x1, dev_x2, dev_l, dev_y = vectorize_window(dev_examples, word_dict, entity_dict, max_windows=max_windows, win=win)
        test_x1, test_x2, test_l, test_y = vectorize_window(test_examples, word_dict, entity_dict, max_windows=max_windows, win=win)

    else:
        train_x1, train_x2, train_l, train_y = vectorize(train_examples, word_dict, entity_dict, sent_len, sent_numb)
        dev_x1, dev_x2, dev_l, dev_y = vectorize(dev_examples, word_dict, entity_dict, sent_len, sent_numb)
        test_x1, test_x2, test_l, test_y = vectorize(test_examples, word_dict, entity_dict, sent_len, sent_numb)

    # plot_dist(train_y,dev_y,test_y)
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

def plot_dist(train_y,dev_y,test_y):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times-Roman')
    sns.set_style(style='white')
    color = sns.color_palette("Set2", 10)
    fig = plt.figure(figsize=(8,12))

    ax1 = fig.add_subplot(3, 1, 1)
    # plt.title("Label distribution",fontsize=20)
    sns.distplot(train_y,kde=False,label='Training', hist=True, norm_hist=True,color="blue")
    ax1.set_xlabel("Answer")
    ax1.set_ylabel("Frequency")
    ax1.set_xlim([0,500])
    plt.legend(loc='best')

    ax2 = fig.add_subplot(3, 1, 2)
    sns.distplot(dev_y,kde=False,label='Validation', hist=True, norm_hist=True,color="green")
    ax2.set_xlabel("Answer")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim([0,500])
    plt.legend(loc='best')

    ax3 = fig.add_subplot(3, 1, 3)
    sns.distplot(test_y,kde=False,label='Test', hist=True, norm_hist=True,color="red")
    ax3.set_xlabel("Answer")
    ax3.set_ylabel("Frequency")
    ax3.set_xlim([0,500])
    plt.legend(loc='best')



    plt.savefig('checkpoints/label_dist.pdf', format='pdf', dpi=300)

    plt.show()



def load_data(in_file, max_example=None, relabeling=True):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """
    # special_char = {"!":"","?":"",":":"",";":"","(":"",")":"",".":"","..":"",
    #                 "...":"",",":"","_":"","'":"","*":"","#":"","-":"","--":"",
    #                 "---":"","\xc2\xa9":"","\xc2\xaa":"","\xe2\x82\xac":"",
    #                 "\"":"","^":"","15\xc2\xbd":""}
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
        answer = f.readline().strip().lower()
        document = f.readline().strip().lower()

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


def build_dict(sentences,win=None, max_words=100000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split('\t'):
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
        d_sents = d.split('\t.\t')
        for i,s in enumerate(d_sents):
            d_sents[i]= s.split('\t')
        # stat_len.append(len(d_sents))
        # stat_wordxsent.append(max([len(s)for s in d_sents]))
        # d_words = d.split(' ')
        q_words = q.split('\t')
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
    #     return sorted(range(len(flatten(seq))), key=lambda x: len(flatten(seq)[x]))
    #
    # if sort_by_len:
    #     # sort by the document length
    #     sorted_index = len_argsort(in_x1)
    #     in_x1 = [in_x1[i] for i in sorted_index]
    #     in_x2 = [in_x2[i] for i in sorted_index]
    #     in_l = in_l[sorted_index]
    #     in_y = [in_y[i] for i in sorted_index]

    return np.array(in_x1), np.expand_dims(np.array(in_x2), axis=1), in_l, np.array(in_y)



def vectorize_window(examples, word_dict, entity_dict, max_windows, win):
    """
        Vectorize `examples` generating windows around candidates
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(examples[0]), len(entity_dict)))
    in_y = []

    #stat_len =[]
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        d_windows = []
        ## vectorize_window document
        d_words = d.split(' ')
        q_words = q.split(' ')
        for j in range(win):
            d_words.insert(0,'<begin-' + str(j) +'>')
            d_words.append('<end-' + str(j) +'>')
            q_words.insert(0,'<begin-' + str(j) +'>')
            q_words.append('<end-' + str(j) +'>')

        assert (a in d_words)
        for i in range(win,len(d_words)-win):
            if(d_words[i] in entity_dict):
                d_windows.append([word_dict[w] if w in word_dict else 0 for w in d_words[i-win:i+win+1]])

        #stat_len.append(len(d_windows))
        # pad to max_windows
        lm = max(0, max_windows - len(d_windows))
        for _ in range(lm):
            d_windows.append([0] * ((win*2)+1))
        d_windows = d_windows[:max_windows]

        for i in range(win,len(q_words)-win):
            if(q_words[i] == '@placeholder'):
                q_windows = [word_dict[w] if w in word_dict else 0 for w in q_words[i-win:i+win+1]]
        in_x1.append(d_windows)
        in_x2.append(q_windows)
        in_l[idx, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0
        in_y.append(entity_dict[a] if a in entity_dict else 0)

    #logging.info('Max sent:{}\t Avg sent: {} Std sent:{}'.format(max(stat_len),sum(stat_len)/len(stat_len),np.std(stat_len)))


    return np.array(in_x1), np.expand_dims(np.array(in_x2), axis=1), in_l, np.array(in_y)


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
