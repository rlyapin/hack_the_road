# The master script to train frame autoencoder

import tensorflow as tf
import numpy as np
import cv2

class FrameAutoencoder:
    # The class to implment frame autoencoder

    def __init__(self, learning_rate=0.1):
        self.session = tf.Session()
        self.learning_rate = learning_rate

        self.frame = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        self.compression = tf.layers.dense(self.frame, units=2, use_bias=False)
        self.restored_frame = tf.layers.dense(self.compression, units=10, use_bias=False)

        # Picking the l2 loss beteen original and restored frames
        self.loss = tf.reduce_mean(tf.pow(self.frame - self.restored_frame, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def compress(self, frames):
        return self.session.run(self.compression, feed_dict={self.frame: frames})

    def restore(self, frames):
        return self.session.run(self.restored_frame, feed_dict={self.frame: frames})

    def loss(self, frames):
        return self.session.run(self.loss, feed_dict={self.frame: frames})

    def train_op(self, frames):
        self.session.run(self.optimizer, feed_dict={self.frame: frames})

test_data = np.arange(10).reshape(1, 10)
encoder_model = FrameAutoencoder(0.1)

print "Loss before training"
print encoder_model.loss(test_data)

print "Started training"
for i in range(100):
    encoder_model.train_op(test_data)

print "Loss after training:"
print encoder_model.loss(test_data)

