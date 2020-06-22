import tensorflow as tf
import numpy as np


# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as GraphSAGE

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# 一种初始化方法
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    # shape[0] = 16384
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    # shape=tf.convert_to_tensor([shape[0],shape[1]])
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
