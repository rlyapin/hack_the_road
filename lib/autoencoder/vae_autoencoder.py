# The script with the definition of frame autoencoder model

import tensorflow as tf
import numpy as np
import cv2

IMAGE_SHAPE = [224, 224, 3]

class FrameAutoencoder:
    # The class to implment variational frame autoencoder

    def __init__(self, session, learning_rate=0.001):
        self.session = session
        self.learning_rate = learning_rate
        self.latent_loss_scaling = 0.001

        # Getting a batch of frames to process
        self.frame = tf.placeholder("float", shape=[None, 224, 224, 3])

        # Adding a collection of convolutional layers to reduce dimensioanlity of compression layer
        conv_1 = tf.layers.conv2d(self.frame, filters=16, kernel_size=[5, 5], 
                                  strides = 1, padding="same", activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2)

        conv_2 = tf.layers.conv2d(pool_1, filters=32, kernel_size=[5, 5], 
                                  strides = 1, padding="same", activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)

        conv_3 = tf.layers.conv2d(pool_2, filters=64, kernel_size=[5, 5], 
                                  strides = 1, padding="same", activation=tf.nn.relu)
        pool_3 = tf.layers.max_pooling2d(conv_3, pool_size=2, strides=2)

        encoder_flat_1 = tf.contrib.layers.flatten(pool_3)
        encoder_flat_2 = tf.layers.dense(encoder_flat_1, 2000, activation=tf.nn.relu)

        # Determining parameters of the latent space (Denoted by z and assumed to be gaussian)
        z_mean = tf.layers.dense(encoder_flat_2, 100)
        z_log_sigma_sq = tf.layers.dense(encoder_flat, 100)

        eps = tf.random_normal(tf.shape(z_mean), 0, 1)
        z = z_mean + eps * tf.sqrt(tf.exp(z_log_sigma_sq))

        # Separately specifying compression
        self.compression = tf.contrib.layers.flatten(z_mean)

        # 28 * 28 * 32 comes from the shape of final pool layer
        decoder_flat_2 = tf.layers.dense(z, 2000, activation=tf.nn.relu)
        decoder_flat_1 = tf.layers.dense(decoder_flat_2, 28 * 28 * 64, activation=tf.nn.relu)

        # Adding a collection of deconvolutions to map the shape of initial input
        # Reversing pooling layer is achieved via deconvolution with more filters (x4) and reshaping
        depool_3 = tf.reshape(decoder_flat_1, tf.shape(pool_3))
        deconv_3 = tf.layers.conv2d_transpose(depool_3, filters=128, kernel_size=[5, 5], 
                                              strides = 1, padding="same", activation=tf.nn.relu)

        depool_2 = tf.reshape(deconv_3, tf.shape(pool_2))
        deconv_2 = tf.layers.conv2d_transpose(depool_2, filters=64, kernel_size=[5, 5], 
                                              strides = 1, padding="same", activation=tf.nn.relu)

        depool_1 = tf.reshape(deconv_2, tf.shape(pool_1))
        deconv_1 = tf.layers.conv2d_transpose(depool_1, filters=12, kernel_size=[5, 5], 
                                              strides = 1, padding="same")

        # Restoring the original batch of frames
        self.restored_frame = tf.reshape(deconv_1, tf.shape(self.frame))

        # Picking the l2 loss beteen original and restored frames
        self.l2_loss = 0.5 * tf.reduce_mean(tf.pow(self.frame - self.restored_frame, 2))
        self.latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq))
        self.vae_loss = tf.minimum(self.l2_loss + self.latent_loss_scaling * self.latent_loss, 10**8)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.vae_loss)

        self.session.run(tf.global_variables_initializer())

    def compress(self, frames):
        return self.session.run(self.compression, feed_dict={self.frame: frames})

    def restore(self, frames):
        return self.session.run(self.restored_frame, feed_dict={self.frame: frames})

    def show_loss(self, frames):
        return self.session.run(self.vae_loss, feed_dict={self.frame: frames})

    def train_op(self, frames):
        self.session.run(self.optimizer, feed_dict={self.frame: frames})
