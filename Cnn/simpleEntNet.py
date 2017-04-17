from __future__ import print_function
from mem_cell import DynamicMemoryCell
import numpy as np
import tensorflow as tf
from functools import partial
from src.utils import variable_summaries


def entNet(sent_numb,num_blocks,embedding_size,learning_rate,clip_gradients,opt,debug=False):
    entity_num = 550
    S_input    = tf.placeholder(tf.float32, shape=[None, sent_numb, embedding_size])
    Q_input    = tf.placeholder(tf.float32, shape=[None, 1, embedding_size])
    A_input    = tf.placeholder(tf.int32, shape=[None, entity_num])
    batch_size = tf.shape(S_input)[0]
    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    with tf.variable_scope('EntityNetwork', initializer=normal_initializer):
        # embeddings = tf.get_variable('word_emb',[vocab_size, embedding_size])
        # story_embedding = tf.nn.embedding_lookup(embeddings, S_input)
        # query_embedding = tf.nn.embedding_lookup(embeddings, Q_input)

        ### sum up all the embeddings in the sentence
        # story_encoded = tf.reduce_sum(story_embedding, reduction_indices=[2])
        # q = tf.reduce_sum(query_embedding, reduction_indices=[2])
        q = Q_input
        ## ENTITY KEYS
        keys = [tf.get_variable('key_{}'.format(j), [embedding_size]) for j in range(num_blocks)]

        ## MEM CELL
        cell = DynamicMemoryCell(num_blocks, embedding_size, keys,
                                        initializer=normal_initializer,
                                        activation=tf.nn.relu)

        ## INITIAL STATE
        initial_state = cell.zero_state(batch_size, tf.float32)
        ## to input into a dynacmicRNN we need to specify the lenght of each sentence
        # used = tf.sign(tf.reduce_max(tf.abs(S_input), reduction_indices=2))
        # length = tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.float32)


        output, last_state = tf.nn.dynamic_rnn(cell, S_input,
                                        initial_state=initial_state)
                                        # sequence_length=length,
        ## split back the final RNN state (in its memory blocks (20)) => [batch,#block,d]
        h_j = tf.pack(tf.split(1, num_blocks, last_state), axis=1)


        ### p_j= qt * h_j  ==> scoring the pair (mem (h_j), question)
        ###  qt * h_j
        temp = tf.reduce_sum(h_j * q, reduction_indices=[2])
        # Subtract max for numerical stability (softmax is shift invariant)
        temp_max = tf.reduce_max(temp, reduction_indices=[-1], keep_dims=True)
        p = tf.nn.softmax(temp - temp_max)


        ### tf.expand_dims(p, 2) because p is (batch_size,#blocks) e
        ### h_j invece (batch_size,#blocks,d) so we need to add a new dim
        ### u is an embedding which represent the wall sentences and facts
        u = tf.reduce_sum(h_j*tf.expand_dims(p, 2), reduction_indices=[1])


        R = tf.get_variable('R', [embedding_size, entity_num])
        H = tf.get_variable('H', [embedding_size, embedding_size])


        ## original q ==>(batch_size,1,d), now I need it just(batch_size,d)
        q = tf.squeeze(q, squeeze_dims=[1])

        ### y has got the size (1,vocab_size)
        y = tf.matmul(tf.nn.relu(q + tf.matmul(u, H)), R)


        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(A_input, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        prediction = tf.argmax(y, 1)
        # loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=A_input,logits=y))
        loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=A_input,logits=y))


        global_step = tf.contrib.framework.get_or_create_global_step()
        #train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        train_op = tf.contrib.layers.optimize_loss(loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=opt,
        clip_gradients=clip_gradients)
        if debug:
            tf.scalar_summary('accuracy', accuracy)
            # tf.scalar_summary('loss', loss)
            tf.histogram_summary('y',y)

            tf.histogram_summary('out',output)
            tf.histogram_summary('last_state',last_state)

            variable_summaries(R,'R')
            variable_summaries(H,'H')


    return accuracy,prediction,loss,train_op,S_input,Q_input, A_input
