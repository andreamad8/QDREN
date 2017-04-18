from __future__ import print_function
from DMC import DynamicMemoryCell
import numpy as np
import tensorflow as tf
from tflearn.activations import sigmoid, softmax

class EntityNetwork():
    def __init__(self, vocab_size, sent_len, sent_numb, num_blocks, embedding_size,
                 learning_rate, decay_steps, decay_rate, opt,
                 trainable, no_out,max_norm, clip_gradients=40.0,embeddings_mat =[],
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
        ## setup placeholder
        self.S = tf.placeholder(tf.int32, shape=[None,self.sent_numb,self.sent_len],name="Story")
        self.Q = tf.placeholder(tf.int32, shape=[None,1,self.sent_len],        name="Question")
        self.A = tf.placeholder(tf.int64, shape=[None],        name="Answer")
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
        correct_prediction = tf.equal(tf.argmax(self.logits, 1),self.A)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")



    def instantiate_weights(self):
        """
        Instantiate Network Weights, including all weights for the Input Encoder, Dynamic
        Memory Cell, as well as Output Decoder.
        """
        ## use pretrained emb
        if (self.trainable[0]):
            E = tf.get_variable('word_emb',[self.vocab_size, self.embedding_size],
                        initializer=tf.constant_initializer(np.array(self.embeddings_mat)),
                        trainable=self.trainable[1])
        else:
            E = tf.get_variable("Embedding",[self.vocab_size, self.embedding_size], initializer=self.init)

        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)],dtype=tf.float32, shape=[self.vocab_size, 1])
        self.E = E * zero_mask

        # Create Learnable Mask
        self.story_mask = tf.get_variable("Story_Mask", [self.sent_len, 1], initializer=tf.constant_initializer(1.0))
        self.query_mask = tf.get_variable("Query_Mask", [self.sent_len, 1], initializer=tf.constant_initializer(1.0))

        # Create Memory Cell Keys
        if (self.trainable[2]):
            self.keys = [tf.get_variable('key_{}'.format(j), [self.embedding_size],
                        initializer=tf.constant_initializer(np.array(self.embeddings_mat[j])),
                        trainable=self.trainable[3]) for j in range(self.num_blocks)]
        else:
            self.keys = [tf.get_variable('key_{}'.format(j), [self.embedding_size]) for j in range(self.num_blocks)]

        # Create Memory Cell
        self.cell = DynamicMemoryCell(self.num_blocks, self.embedding_size, self.keys)

        if (not self.no_out):
            # Output Module Variables
            self.H = tf.get_variable("H", [self.embedding_size, self.embedding_size], initializer=self.init)
            self.R = tf.get_variable("R", [self.embedding_size, self.vocab_size], initializer=self.init)


    def inference(self):
        """
        Build inference pipeline, going from the story and question, through the memory cells, to the
        distribution over possible answers.
        """

        # Story Input Encoder
        story_embeddings = tf.nn.embedding_lookup(self.E, self.S,max_norm=self.max_norm)             # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.multiply(story_embeddings, self.story_mask)     # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.reduce_sum(story_embeddings, axis=[2])          # Shape: [None, story_len, embed_sz]

        # Query Input Encoder
        query_embedding = tf.nn.embedding_lookup(self.E, self.Q,max_norm=self.max_norm)              # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.multiply(query_embedding, self.query_mask)       # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.reduce_sum(query_embedding, axis=[2])            # Shape: [None, embed_sz]

        ## to input into a dynacmicRNN we need to specify the lenght of each sentence
        length = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(self.S), axis=2)), axis=1), tf.int32)

        # Send Story through Memory Cell
        initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        _, memories = tf.nn.dynamic_rnn(self.cell, story_embeddings,
                                        sequence_length=length,
                                        initial_state=initial_state)

        # Output Module
        stacked_memories = tf.stack(memories, axis=1)


        # Generate Memory Scores
        p_scores = softmax(tf.reduce_sum(tf.multiply(stacked_memories,        # Shape: [None, mem_slots]
                                                      query_embedding), axis=[2]))

        # Subtract max for numerical stability (softmax is shift invariant)
        p_max = tf.reduce_max(p_scores, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(p_scores - p_max)
        attention = tf.expand_dims(attention, 2)                              # Shape: [None, mem_slots, 1]

        if (self.no_out):
            return p_scores
        else:
            # Weight memories by attention vectors
            u = tf.reduce_sum(tf.multiply(stacked_memories, attention), axis=1)   # Shape: [None, embed_sz]

            # Output Transformations => Logits

            hidden = tf.nn.relu(tf.matmul(u, self.H) + tf.squeeze(query_embedding))                # Shape: [None, embed_sz]
            logits = tf.matmul(hidden, self.R)
            return logits

    def loss(self):
        """
        Build loss computation - softmax cross-entropy between logits, and correct answer.
        """
        return tf.losses.sparse_softmax_cross_entropy(self.A, self.logits)

    def train(self):
        """
        Build Optimizer Training Operation.
        """
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)

        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer=self.opt,
                                                   clip_gradients=self.clip_gradients)
        return train_op
