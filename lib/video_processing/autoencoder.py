# The script with the definition of frame autoencoder model

import tensorflow as tf
import numpy as np
import cv2

IMAGE_SHAPE = [224, 224, 3]

class FrameAutoencoder:
    # The class to implment variational frame autoencoder

    def __init__(self, session, learning_rate=0.1):
        self.session = session
        self.learning_rate = learning_rate

        # Getting a batch of frames to process
        self.frame = tf.placeholder("float", shape=[None, 224, 224, 3])

        # Adding a collection of convolutional layers to reduce dimensioanlity of compression layer
        conv_1 = tf.layers.conv2d(self.frame, filters=16, kernel_size=[5, 5], 
                                  strides = 2, padding="same", activation=tf.nn.relu)
        conv_2 = tf.layers.conv2d(conv_1, filters=8, kernel_size=[5, 5], 
                                  strides = 2, padding="same", activation=tf.nn.relu)
        conv_3 = tf.layers.conv2d(conv_2, filters=4, kernel_size=[5, 5], 
                                  strides = 2, padding="same", activation=tf.nn.relu)
        conv_4 = tf.layers.conv2d(conv_3, filters=2, kernel_size=[5, 5], 
                                  strides = 2, padding="same", activation=tf.nn.relu)

        # Determining parameters of the latent space (Denoted by z and assumed to be gaussian)
        z_mean = tf.layers.conv2d(conv_4, filters=1, kernel_size=[5, 5], 
                                  strides = 2, padding="same", activation=tf.nn.relu)

        z_log_sigma_sq = tf.layers.conv2d(conv_4, filters=1, kernel_size=[5, 5], 
                                          strides = 2, padding="same", activation=tf.nn.relu)

        eps = tf.random_normal(tf.shape(z_mean), 0, 1)
        z = z_mean + eps * tf.sqrt(tf.exp(z_log_sigma_sq))

        # Currently for [224 , 224] images compression layer has 49 elements
        self.compression = tf.contrib.layers.flatten(z_mean)

        # Adding a collection of deconvolutions to map the shape of initial input
        deconv_5 = tf.layers.conv2d_transpose(z, filters=2, kernel_size=[5, 5], 
                                              strides = 2, padding="same", activation=tf.nn.relu)
        deconv_4 = tf.layers.conv2d_transpose(deconv_5, filters=4, kernel_size=[5, 5], 
                                              strides = 2, padding="same", activation=tf.nn.relu)
        deconv_3 = tf.layers.conv2d_transpose(deconv_4, filters=8, kernel_size=[5, 5], 
                                              strides = 2, padding="same", activation=tf.nn.relu)
        deconv_2 = tf.layers.conv2d_transpose(deconv_3, filters=16, kernel_size=[5, 5], 
                                              strides = 2, padding="same", activation=tf.nn.relu)

        # Restoring the original batch of frames
        self.restored_frame = tf.layers.conv2d_transpose(deconv_2, filters=3, kernel_size=[5, 5], 
                                                         strides = 2, padding="same", activation=tf.nn.relu)

        # Picking the l2 loss beteen original and restored frames
        l2_loss = tf.reduce_mean(tf.pow(self.frame - self.restored_frame, 2))
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq))
        self.vae_loss = l2_loss + latent_loss
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.vae_loss)

        self.session.run(tf.global_variables_initializer())

    def compress(self, frames):
        return self.session.run(self.compression, feed_dict={self.frame: frames})

    def restore(self, frames):
        return self.session.run(self.restored_frame, feed_dict={self.frame: frames})

    def loss(self, frames):
        return self.session.run(self.vae_loss, feed_dict={self.frame: frames})

    def train_op(self, frames):
        self.session.run(self.optimizer, feed_dict={self.frame: frames})
