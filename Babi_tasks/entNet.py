from __future__ import print_function
from mem_cell import DynamicMemoryCell
import numpy as np
import tensorflow as tf
from functools import partial
from src.utils import variable_summaries


def entNet(vocab_size,sent_len,sent_numb,num_blocks,embeddings_mat,embedding_size,learning_rate,clip_gradients,opt,debug):
    S_input = tf.placeholder(tf.int32, shape=[None,sent_numb,sent_len],name="Story")
    Q_input = tf.placeholder(tf.int32, shape=[None,1,sent_len],name="Question")
    A_input = tf.placeholder(tf.int32, shape=[None,vocab_size], name="Answer")
    batch_size = tf.shape(S_input)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)

    with tf.variable_scope('EntityNetwork', initializer=normal_initializer):
        # embeddings = tf.get_variable('word_emb',[vocab_size, embedding_size])
        embeddings = tf.get_variable('word_emb',[vocab_size, embedding_size],
                                     initializer=tf.constant_initializer(np.array(embeddings_mat)),
                                     trainable=True)

        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],dtype=tf.float32, shape=[vocab_size, 1])
        embedding_params_masked = embeddings * zero_mask


        # Create Learnable Mask
        story_mask = tf.get_variable("Story_Mask", [sent_len, 1], initializer=tf.constant_initializer(1.0))
        story_embedding = tf.nn.embedding_lookup(embeddings, S_input, max_norm=1)
        story_embedding = tf.multiply(story_embedding, story_mask)     # Shape: [None, story_len, sent_len, embed_sz]
        story_encoded = tf.reduce_sum(story_embedding, axis=[2])


        query_mask = tf.get_variable("Query_Mask", [sent_len, 1], initializer=tf.constant_initializer(1.0))
        query_embedding = tf.nn.embedding_lookup(embeddings, Q_input, max_norm=1)
        query_embedding = tf.multiply(query_embedding, query_mask)       # Shape: [None, sent_len, embed_sz]
        q = tf.reduce_sum(query_embedding, axis=[2])

        # ### sum up all the embeddings in the sentence
        # story_encoded = tf.reduce_sum(story_embedding, axis=[2])
        # story_encoded = tf.nn.l2_normalize(story_encoded,-1)
        # q = tf.reduce_sum(query_embedding, axis=[2])
        # q = tf.nn.l2_normalize(q,-1)

        ## ENTITY KEYS
        # keys = [tf.get_variable('key_{}'.format(j), [embedding_size]) for j in range(num_blocks)]
        keys = [tf.get_variable('key_{}'.format(j), [embedding_size],
                                initializer=tf.constant_initializer(np.array(embeddings_mat[j])),
                                trainable=True) for j in range(num_blocks)]


        ## MEM CELL
        cell = DynamicMemoryCell(num_blocks, embedding_size, keys)
        # ,
        #                                 initializer=normal_initializer,
        #                                 activation=tf.nn.relu)

        ## INITIAL STATE
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        ## to input into a dynacmicRNN we need to specify the lenght of each sentence
        used = tf.sign(tf.reduce_max(tf.abs(S_input), axis=2))
        length = tf.cast(tf.reduce_sum(used, axis=1), tf.int32)


        output, last_state = tf.nn.dynamic_rnn(cell, story_encoded,
                                        sequence_length=length,
                                        initial_state=initial_state)
        ## split back the final RNN state (in its memory blocks (20)) => [batch,#block,d]
        # h_j = tf.stack(tf.split(1, num_blocks, last_state), axis=1)
        h_j = tf.stack(last_state, axis=1)

        ### p_j= qt * h_j  ==> scoring the pair (mem (h_j), question)
        ###  qt * h_j
        temp = tf.reduce_sum(tf.multiply(h_j,q), axis=[2])
        # Subtract max for numerical stability (softmax is shift invariant)
        # temp_max = tf.reduce_max(temp, axis=[-1], keep_dims=True)
        # p = tf.nn.softmax(temp - temp_max)
        y = tf.nn.softmax(temp)


        # ### tf.expand_dims(p, 2) because p is (batch_size,#blocks) e
        # ### h_j invece (batch_size,#blocks,d) so we need to add a new dim
        # ### u is an embedding which represent the wall sentences and facts
        # u = tf.reduce_sum(h_j*tf.expand_dims(p, 2), axis=[1])
        #
        #
        # R = tf.get_variable('R', [embedding_size, vocab_size])
        # H = tf.get_variable('H', [embedding_size, embedding_size])
        #
        #
        # ## original q ==>(batch_size,1,d), now I need it just(batch_size,d)
        # q = tf.squeeze(q, squeeze_dims=[1])
        #
        # ### y has got the size (1,vocab_size)
        # y = tf.matmul(tf.nn.relu(q + tf.matmul(u, H)), R)


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
            tf.summary.scalar('accuracy', accuracy)
            # tf.scalar_summary('loss', loss)
            tf.summary.histogram('y',y)
            # tf.histogram_summary('gates',gates)
            # tf.histogram_summary('out',output)
            # tf.histogram_summary('last_state',last_state)

            # variable_summaries(R,'R')
            # variable_summaries(H,'H')
            # variable_summaries(embeddings,'embeddings')





    return accuracy,prediction,loss,train_op,S_input,Q_input,A_input,output
