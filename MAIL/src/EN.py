from __future__ import print_function
from memories.DMC_query import DynamicMemoryCell
import numpy as np
import tensorflow as tf
from tflearn.activations import sigmoid, softmax
from functools import partial
import logging


class EntityNetwork():
    def __init__(self, vocab_size, sent_len, sent_numb, num_blocks, embedding_size,
                 learning_rate, decay_steps, decay_rate, opt, label_num,L2,
                 trainable, no_out,max_norm, clip_gradients ,embeddings_mat =[],
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Initialize an Entity Network with the necessary hyperparameters.

        :param vocabulary: Word Vocabulary for given model.
        :param sentence_len: Maximum length of a sentence.
        :param story_len: Maximum length of a story.
        """
        self.vocab_size, self.sent_len, self.sent_numb = vocab_size, sent_len, sent_numb
        self.embedding_size, self.num_blocks, self.init = embedding_size, num_blocks, initializer
        self.opt, self.embeddings_mat, self.trainable = opt, embeddings_mat, trainable
        self.clip_gradients, self.no_out, self.max_norm = clip_gradients, no_out,max_norm
        self.learning_rate ,self.decay_steps, self.decay_rate= learning_rate,decay_steps,decay_rate
        self.label_num,  self.L2 = label_num, L2
        ## setup placeholder
        self.S = tf.placeholder(tf.int32, shape=[None,self.sent_numb,self.sent_len],name="Story")
        self.Q = tf.placeholder(tf.int32, shape=[None,1,self.sent_len],        name="Question")
        self.A = tf.placeholder(tf.int64, shape=[None],        name="Answer")
        # self.C = tf.placeholder(tf.int32, shape=[None,self.label_num],name="Candidate")

        self.keep_prob = tf.placeholder(tf.float32, name= "dropout")

        self.batch_size = tf.shape(self.S)[0]

        # Setup Global, Epoch Step
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # Instantiate Network Weights
        self.instantiate_weights()

        # Build Inference Pipeline
        self.logits = self.inference()

        # Build Loss Computation
        self.loss_val = self.loss()

        # Build Training Operation
        self.train_op = self.train()

        # Create operations for computing the accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1),self.A)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="Accuracy")

        parameters = self.count_parameters()
        logging.info('Parameters: {}'.format(parameters))

    def instantiate_weights(self):
        """
        Instantiate Network Weights, including all weights for the Input Encoder, Dynamic
        Memory Cell, as well as Output Decoder.
        """
        ## use pretrained emb
        if (self.trainable[0]):
            # E = tf.get_variable('word_emb',self.embeddings_mat)
            self.E = tf.get_variable('Embedding',[self.vocab_size, self.embedding_size],
                        initializer=tf.constant_initializer(np.array(self.embeddings_mat)),
                        trainable=self.trainable[1])
        else:
            self.E = tf.get_variable("Embedding",[self.vocab_size, self.embedding_size], initializer=self.init)
        # 
        # zero_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)],dtype=tf.float32, shape=[self.vocab_size, 1])
        # self.E = self.E * zero_mask

        # alpha = tf.get_variable(name='alpha',
        #                         shape=self.embedding_size,
        #                         initializer=tf.constant_initializer(1.0))
        # self.activation = partial(prelu, alpha=alpha)

        # Create Learnable Mask
        self.story_mask = tf.get_variable("Story_Mask", [self.sent_len, self.embedding_size],
                                          initializer=tf.constant_initializer(1.0),trainable=True)
        self.query_mask = tf.get_variable("Query_Mask", [self.sent_len, self.embedding_size],
                                          initializer=tf.constant_initializer(1.0),trainable=True)

        # Create Memory Cell Keys
        # if (self.trainable[2]):
        #     # candidate_E = tf.nn.embedding_lookup(self.E,self.C,max_norm=self.max_norm)
        #     self.keys = [tf.get_variable('key_{}'.format(j), [self.embedding_size],
        #                 initializer=tf.constant_initializer(np.array(self.embeddings_mat[j])),
        #                 trainable=self.trainable[3]) for j in range(self.num_blocks)]
        # else:
        self.keys = [tf.get_variable('key_{}'.format(j), [self.embedding_size]) for j in range(self.num_blocks)]


        # Output Module Variables
        self.H = tf.get_variable("H", [self.embedding_size, self.embedding_size], initializer=self.init)
        self.R = tf.get_variable("R", [self.embedding_size, self.label_num], initializer=self.init)


    def inference(self):
        """
        Build inference pipeline, going from the story and question, through the memory cells, to the
        distribution over possible answers.
        """

        # Story Input Encoder
        story_embeddings = tf.nn.embedding_lookup(self.E, self.S,max_norm=self.max_norm) # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.nn.dropout(story_embeddings, self.keep_prob)               # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.multiply(story_embeddings, self.story_mask)
        story_embeddings = tf.reduce_sum(story_embeddings, axis=[2])                     # Shape: [None, story_len, embed_sz]

        # Query Input Encoder
        query_embedding = tf.nn.embedding_lookup(self.E, self.Q,max_norm=self.max_norm)  # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.multiply(query_embedding, self.query_mask)                  # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.reduce_sum(query_embedding, axis=[2])                       # Shape: [None, 1, embed_sz]

        ## to input into a dynacmicRNN we need to specify the lenght of each sentence
        # length = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(self.S), axis=2)), axis=1), tf.int32)
        self.length = self.get_sequence_length()

        # Create Memory Cell
        self.cell = DynamicMemoryCell(self.num_blocks, self.embedding_size,
                                      self.keys, query_embedding)
        # self.cell =tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)

        # Send Story through Memory Cell
        initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        self.out, memories = tf.nn.dynamic_rnn(self.cell, story_embeddings,
                                        sequence_length=self.length,
                                        initial_state=initial_state)

        # Output Module
        # stacked_memories = tf.stack(memories, axis=1)
        stacked_memories = tf.stack(tf.split(memories, self.num_blocks, 1), 1)


        # Generate Memory Scores
        p_scores = softmax(tf.reduce_sum(tf.multiply(stacked_memories,                   # Shape: [None, mem_slots]
                                                      query_embedding), axis=[2]))


        if (self.no_out):
            u = tf.reduce_max(stacked_memories,axis=1,keep_dims=False)
            # u = tf.reduce_mean(stacked_memories,axis=1,keep_dims=False)
        else:
            # Subtract max for numerical stability (softmax is shift invariant)
            p_max = tf.reduce_max(p_scores, axis=-1, keep_dims=True)
            attention = tf.nn.softmax(p_scores - p_max)
            attention = tf.expand_dims(attention, 2)                                         # Shape: [None, mem_slots, 1]

            # Weight memories by attention vectors
            u = tf.reduce_sum(tf.multiply(stacked_memories, attention), axis=1)          # Shape: [None, embed_sz]

        # Output Transformations => Logits
        hidden = sigmoid(tf.matmul(u, self.H) + tf.squeeze(query_embedding))      # Shape: [None, embed_sz]
        logits = tf.matmul(hidden, self.R)
        return logits

    def loss(self):
        """
        Build loss computation - softmax cross-entropy between logits, and correct answer.
        """
        if(self.L2 !=0.0):
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'rnn/DynamicMemoryCell/biasU:0' != v.name ])  * self.L2
            # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var])  * self.L2
            return tf.losses.sparse_softmax_cross_entropy(self.A,self.logits)+lossL2
        else:
            return tf.losses.sparse_softmax_cross_entropy(self.A,self.logits)

    def train(self):
        """
        Build Optimizer Training Operation.
        """
        # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
        #                                            self.decay_rate, staircase=True)

        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.learning_rate, optimizer=self.opt,
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def get_sequence_length(self):
        """
        This is a hacky way of determining the actual length of a sequence that has been padded with zeros.
        """
        used = tf.sign(tf.reduce_max(tf.abs(self.S), axis=-1))
        length = tf.cast(tf.reduce_sum(used, axis=-1), tf.int32)
        return length

    def position_encoding(self,sentence_size, embedding_size):
        """
        Position Encoding described in section 4.1 [1]
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size+1
        le = embedding_size+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        return np.transpose(encoding)

    def count_parameters(self):
        "Count the number of parameters listed under TRAINABLE_VARIABLES."
        num_parameters = sum([np.prod(tvar.get_shape().as_list())
                              for tvar in tf.trainable_variables()])
        return num_parameters

def prelu(features, alpha, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU'):
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
