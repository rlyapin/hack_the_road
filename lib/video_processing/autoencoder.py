# The script with the definition of frame autoencoder model

import tensorflow as tf
import numpy as np
import cv2
from tf_saver import TfSaver

class FrameAutoencoder:
    # The class to implment frame autoencoder

    def __init__(self, session, learning_rate=0.1):
        self.session = session
        self.learning_rate = learning_rate

        self.frame = tf.placeholder("float", shape=[None, 10])
        self.compression = tf.layers.dense(self.frame, units=2, use_bias=False)
        self.restored_frame = tf.layers.dense(self.compression, units=10, use_bias=False)

        # Picking the l2 loss beteen original and restored frames
        self.l2loss = tf.reduce_mean(tf.pow(self.frame - self.restored_frame, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2loss)

        self.session.run(tf.global_variables_initializer())

    def compress(self, frames):
        return self.session.run(self.compression, feed_dict={self.frame: frames})

    def restore(self, frames):
        return self.session.run(self.restored_frame, feed_dict={self.frame: frames})

    def loss(self, frames):
        return self.session.run(self.l2loss, feed_dict={self.frame: frames})

    def train_op(self, frames):
        self.session.run(self.optimizer, feed_dict={self.frame: frames})
