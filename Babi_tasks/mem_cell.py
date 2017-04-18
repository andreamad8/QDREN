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
            address_g = tf.reduce_sum(tf.multiply(inputs,
                                      tf.expand_dims(self.keys[block_id], 0)), axis=[1]) # Shape: [bsz]
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
#
# class DynamicMemoryCell(tf.nn.rnn_cell.RNNCell):
#     """
#     Implementation of a dynamic memory cell as a gated recurrent network.
#     The cell's hidden state is divided into blocks and each block's weights are tied.
#     """
#     def __init__(self, num_blocks, num_units_per_block, keys, initializer, activation=tf.nn.relu,debug= False):
#         self._num_blocks = num_blocks # M
#         self._num_units_per_block = num_units_per_block # d
#         self._keys = keys
#         self._activation = activation # \phi
#         self._initializer = initializer
#         self._debug = debug
#
#     @property
#     def state_size(self):
#         return self._num_blocks * self._num_units_per_block
#
#     @property
#     def output_size(self):
#         return self._num_blocks * self._num_units_per_block
#
#     def zero_state(self, batch_size, dtype):
#         """
#         We initialize the memory to the key values.
#         """
#
#         zero_state = tf.concat(1, [tf.expand_dims(key, 0) for key in self._keys])
#         zero_state_batch = tf.tile(zero_state, tf.pack([batch_size, 1]))
#         return zero_state_batch
#
#     def get_gate(self, state_j, key_j, inputs,j):
#         """
#         Implements the gate (scalar for each block). Equation 2:
#
#         g_j <- \sigma(s_t^T h_j + s_t^T w_j)
#         """
#         a = tf.reduce_sum(inputs * state_j, reduction_indices=[1])
#         # a = tf.Print(a, [a], message="s_t^T h_j gate"+str(j))
#         b = tf.reduce_sum(inputs * tf.expand_dims(key_j, 0), reduction_indices=[1])
#         # b = tf.Print(b, [b], message="s_t^T w_j gate"+str(j))
#
#         return tf.sigmoid(a + b)
#
#     def get_candidate(self, state_j, key_j, inputs, U, V, W):
#         """
#         Represents the new memory candidate that will be weighted by the
#         gate value and combined with the existing memory. Equation 3:
#
#         h_j^~ <- \phi(U h_j + V w_j + W s_t)
#         """
#         key_V = tf.matmul(tf.expand_dims(key_j, 0), V)
#         state_U = tf.matmul(state_j, U)
#         inputs_W = tf.matmul(inputs, W)
#         return self._activation(state_U + key_V + inputs_W)
#
#
#     def __call__(self, inputs, state, scope=None):
#         with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
#             # Split the hidden state into blocks (each U, V, W are shared across blocks).
#             state = tf.split(1, self._num_blocks, state)
#
#             # TODO: ortho init?
#             U = tf.get_variable('U', [self._num_units_per_block, self._num_units_per_block])
#             V = tf.get_variable('V', [self._num_units_per_block, self._num_units_per_block])
#             W = tf.get_variable('W', [self._num_units_per_block, self._num_units_per_block])
#
#             # TODO: layer norm?
#
#             next_states = []
#             for j, state_j in enumerate(state): # Hidden State (j)
#                 key_j = self._keys[j]
#                 gate_j = self.get_gate(state_j, key_j, inputs,j)
#                 # gate_j = tf.Print(gate_j, [gate_j], message="Gate{}".format(j))
#                 candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W)
#
#                 # Equation 4: h_j <- h_j + g_j * h_j^~
#                 # Perform an update of the hidden state (memory).
#                 state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j
#
#                 # Equation 5: h_j <- h_j / \norm{h_j}
#                 # Forget previous memories by normalization.
#                 state_j_next = tf.nn.l2_normalize(state_j_next, -1, epsilon=1e-7) # TODO: Is epsilon necessary?
#
#                 next_states.append(state_j_next)
#
#             state_next = tf.concat(1, next_states)
#         return state_next, state_next
