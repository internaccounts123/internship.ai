import numpy as np
import tensorflow as tf
import keras.backend as keras


def weighted_cross_entropy_loss(y,x):
    """
    Params:
        x:Tensor of shape(num_examples,num actions). It represents the probabilities
        y:one hot  shape=(num_examples,num actions)
    """
    weightages=KB.sum(y,axis=0,keepdims=True)/ tf.cast(KB.shape(y)[0],tf.float32)
    weightages=weightages*tf.cast(y,tf.float32)
    weightages=KB.sum(weightages,axis=1)
    loss=tf.losses.softmax_cross_entropy(y,x,weights=tf.constant(1.0,dtype=tf.float32)-weightages)
    return loss

