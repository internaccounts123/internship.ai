import numpy as np
import tensorflow as tf
import keras.backend as KB


def weighted_cross_entropy_loss(y, x):
    """
    Params:
       x:Tensor of shape(num_examples,num actions). It represents the probabilities
       y:one hot  shape=(num_examples,num actions)
    """
    weightages = KB.sum(y, axis=0, keepdims=True) / tf.cast(KB.shape(y)[0], tf.float32)
    weightages = weightages * tf.cast(y, tf.float32)
    weightages = KB.sum(weightages, axis=1)
    loss = tf.losses.softmax_cross_entropy(y, x, weights=tf.constant(1.0, dtype=tf.float32) - weightages)
    return loss


def f1_score_metric(num_actions):
    """
    this function returns a metric that returns weighted f1 scores
    """

    def f1_calc(y, x):
        F1_list = []
        pred_labels = tf.cast(tf.argmax(x, axis=1), tf.int32)
        target_labels = tf.cast(tf.argmax(y, axis=1), tf.int32)
        y = tf.cast(y, tf.int32)
        for i in range(num_actions):
            action = tf.constant(i, dtype=tf.int32)
            targets = tf.cast(tf.equal(target_labels, action), tf.float32)
            preds = tf.cast(tf.equal(pred_labels, action), tf.float32)

            TP = tf.count_nonzero(preds * targets, dtype=tf.float32)
            TN = tf.count_nonzero((1 - preds) * (1 - targets), dtype=tf.float32)
            FP = tf.count_nonzero(preds * (1 - targets), dtype=tf.float32)
            FN = tf.count_nonzero((1 - preds) * targets, dtype=tf.float32)

            error = 1e-6
            Pr = (TP) / (TP + FP + error)
            Re = (TP) / (TP + FN + error)
            F1 = (2 * Pr * Re) / (Pr + Re + error)
            F1_list.append(F1)

        weighted_f1 = F1_list[0]
        for f1 in F1_list[1:]:
            weighted_f1 = f1 + weighted_f1

        return weighted_f1 / tf.constant(num_actions, dtype=tf.float32)

    return f1_calc