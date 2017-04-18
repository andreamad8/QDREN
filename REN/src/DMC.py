import numpy as np
import tensorflow as tf
from src.utils import variable_summaries
import tflearn
from collections import namedtuple
from tflearn.activations import sigmoid, softmax
import functools

def prelu_func(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
prelu = functools.partial(prelu_func, initializer=tf.constant_initializer(1.0))

class DynamicMemoryCell(tf.contrib.rnn.RNNCell):
    def __init__(self, memory_slots, memory_size, keys, activation=prelu,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Instantiate a DynamicMemory Cell, with the given number of memory slots, and key vectors.
        :param memory_slots: Number of memory slots to initialize.
        :param memory_size: Dimensionality of memories => tied to embedding size.
        :param keys: List of keys to seed the Dynamic Memory with (can be random).
        :param initializer: Variable Initializer for Cell Parameters.
        """
        self.m, self.mem_sz, self.keys = memory_slots, memory_size, keys
        self.activation, self.init = activation, initializer

        # Instantiate Dynamic Memory Parameters => CONSTRAIN HERE
        self.U = tf.get_variable("U", [self.mem_sz, self.mem_sz], initializer=self.init)
        self.V = tf.get_variable("V", [self.mem_sz, self.mem_sz], initializer=self.init)
        self.W = tf.get_variable("W", [self.mem_sz, self.mem_sz], initializer=self.init)

    @property
    def state_size(self):
        """
        Return size of DynamicMemory State - for now, just M x d.
        """
        return [self.mem_sz for _ in range(self.m)]

    @property
    def output_size(self):
        return [self.mem_sz for _ in range(self.m)]

    def zero_state(self, batch_size, dtype):
        """
        Initialize Memory to start as Key Values
        """
        return [tf.tile(tf.expand_dims(key, 0), [batch_size, 1]) for key in self.keys]

    def __call__(self, inputs, state, scope=None):
        """
        Run the Dynamic Memory Cell on the inputs, updating the memories with each new time step.
        :param inputs: 2D Tensor of shape [bsz, mem_sz] representing a story sentence.
        :param states: List of length M, each with 2D Tensor [bsz, mem_sz] => h_j (starts as key).
        """
        new_states = []
        for block_id, h in enumerate(state):
            # Gating Function
            content_g = tf.reduce_sum(tf.multiply(inputs, h), axis=[1])                  # Shape: [bsz]
            address_g = tf.reduce_sum(tf.multiply(inputs,tf.expand_dims(self.keys[block_id], 0)), axis=[1]) # Shape: [bsz]
            g = sigmoid(content_g + address_g)

            # New State Candidate
            h_component = tf.matmul(h, self.U)                                           # Shape: [bsz, mem_sz]
            w_component = tf.matmul(tf.expand_dims(self.keys[block_id], 0), self.V)      # Shape: [1, mem_sz]
            s_component = tf.matmul(inputs, self.W)                                      # Shape: [bsz, mem_sz]
            candidate = self.activation(h_component + w_component + s_component)         # Shape: [bsz, mem_sz]

            # State Update
            new_h = h + tf.multiply(tf.expand_dims(g, -1), candidate)                    # Shape: [bsz, mem_sz]

            # Unit Normalize State
            new_h_norm = tf.nn.l2_normalize(new_h, -1)                                   # Shape: [bsz, mem_sz]
            new_states.append(new_h_norm)

        return new_states, new_states
