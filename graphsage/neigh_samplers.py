from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

"""
Classes that are used to sample node neighborhoods
"""


class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """

    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)  # embedding_lookup相当于通过key找出对应的张量
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        # if FLAGS.aggregate_with_self:
        #     ids=tf.reshape(ids,[-1,1])
        #     adj_lists=tf.concat([ids,adj_lists],axis=1)
        return adj_lists
