# The script with the definition of the lstm model to detect parking mode

import tensorflow as tf
import numpy as np
import cv2
from tf_saver import TfSaver

# Currently compression size is: 49 entries for autoencoder and 2 entries for optical flow
# MAX_SEQUENCE_LENGTH: we take 0.2s break between for frames and ~10s seconds clips for prediction
# NUM_HIDDEN is the dimensionality of hidden lstm layer
COMPRESSION_SIZE = 51
MAX_SEQUENCE_LENGTH = 50
NUM_HIDDEN = 25

class LSTMClassifier:
    # The class to implment lstm model

    def __init__(self, session, learning_rate=0.1):
        self.session = session
        self.learning_rate = learning_rate

        # Getting a sequence of frames to process
        # Note that here it is implied these frames are already compressed into smaller vector
        self.frames = tf.placeholder("float", shape=[None, MAX_SEQUENCE_LENGTH, COMPRESSION_SIZE])
        self.labels = tf.placeholder("float", shape=[None, 1])

        # Getting LSTM output
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN)
        output, _ = tf.nn.dynamic_rnn(lstm_cell, self.frames)
        final_lstm_state = output[:, -1, :]

        # Getting the probabilities of parking
        pre_prob_layer = tf.layers.dense(final_lstm_state, units=2)
        self.prob_layer = tf.nn.softmax(pre_prob_layer)

        # Defining prediction loss
        one_hot_labels = tf.one_hot(self.labels, depth=2, on_value=0.0, off_value=1.0)
        self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=self.prob_layer)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.pred_loss)

        self.session.run(tf.global_variables_initializer())

    def predict(self, frames):
        return self.session.run(self.prob_layer, feed_dict={self.frames: frames})

    def loss(self, frames):
        return self.session.run(self.pred_loss, feed_dict={self.frames: frames})

    def train_op(self, frames):
        self.session.run(self.optimizer, feed_dict={self.frames: frames})
