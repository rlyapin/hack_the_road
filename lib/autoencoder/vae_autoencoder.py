
# The code to implement a variational autoencoder

import tensorflow as tf
import numpy as np


class VAEAutoencoder:
    def __define_placeholders(self, input_shape):
        """Setting up autoencoder variables"""
        # None in shape stands for the batch size
        self.input_frame = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.latent_loss_weight = tf.placeholder(tf.float32, shape=[])
        self.dropout_share = tf.placeholder(tf.float32, shape=[])


    @staticmethod
    def __add_conv_block(input_layer, n_filters_1, n_filters_2, dropout_share):
        """A helper function to add convolution blocks in encoder part of vae

        Each block here would consist of two convolutional layers with dropout followed by max pooling
        Note that I am using RELU activation for convilutional layers
        This means on its own the block output is a bad fit for the bottleneck vae layer (cannot be negative)

        Args:
            input_layer: a tf layer that is fed into convolution block with [None, None, None, None] shape
            n_filters_1: a number of filters for the first convolution layer
            n_filters_2: a number of filters for the second convolution layer
            dropout_share: a parameter to feed to the dropout layer
        Returns:
            tf layer after convolutions
        """ 
        conv_1 = tf.layers.conv2d(input_layer, filters=n_filters_1, kernel_size=[3, 3], 
                                  strides=1, padding="same", activation=tf.nn.relu)
        drop_1 = tf.nn.dropout(conv_1, keep_prob=1-dropout_share)
        conv_2 = tf.layers.conv2d(drop_1, filters=n_filters_2, kernel_size=[3, 3], 
                                  strides=1, padding="same", activation=tf.nn.relu)
        drop_2 = tf.nn.dropout(conv_2, keep_prob=1-dropout_share)
        pool_1 = tf.layers.max_pooling2d(drop_2, pool_size=2, strides=2)
        return pool_1


    @staticmethod
    def __undo_conv_block(input_layer, n_filters_1, n_filters_2, dropout_share):
        """A helper function to add deconvolution blocks in decoder part of vae

        The function is supposed to be a counterpart to __add_conv_block and repeat actions there in reverse
        Note that unpooling part simply doubles image shape so mismatch with encoder can occur for odd shapes
        RELU activations allow the block output to be a recovered image (if it is originally in [0, 1] range)

        Args:
            input_layer: a tf layer that is fed into deconvolution block with [None, None, None, None] shape
            n_filters_1: a number of filters for the first deconvolution layer
            n_filters_2: a number of filters for the second deconvolution layer
            dropout_share: a parameter to feed to the dropout layer
        Returns:
            tf layer after deconvolutions
        """ 
        unpool_1 = tf.image.resize_images(input_layer, 
                                          2 * np.array(input_layer.get_shape().as_list()[1:3]), 
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv_1 = tf.layers.conv2d_transpose(unpool_1, filters=n_filters_1, kernel_size=[3, 3], 
                                              strides=1, padding="same", activation=tf.nn.relu)
        drop_1 = tf.nn.dropout(deconv_1, keep_prob=1-dropout_share)
        deconv_2 = tf.layers.conv2d_transpose(drop_1, filters=n_filters_2, kernel_size=[3, 3], 
                                              strides=1, padding="same", activation=tf.nn.relu)
        drop_2 = tf.nn.dropout(deconv_2, keep_prob=1-dropout_share)    
        return drop_2


    def __encoder(self, hidden_size):
        """Setting up the encoding part of autoencoder"""
        # Adding several convolution blocks
        self.conv_block_1 = VAEAutoencoder.__add_conv_block(self.input_frame, 16, 32, self.dropout_share)
        self.conv_block_2 = VAEAutoencoder.__add_conv_block(self.conv_block_1, 32, 64, self.dropout_share)
        self.conv_block_3 = VAEAutoencoder.__add_conv_block(self.conv_block_2, 64, 128, self.dropout_share)
        self.conv_block_4 = VAEAutoencoder.__add_conv_block(self.conv_block_3, 128, 256, self.dropout_share)
        self.conv_block_5 = VAEAutoencoder.__add_conv_block(self.conv_block_4, 256, 512, self.dropout_share)

        # Flattening the results and defining a bottleneck layer "z"
        # Note that at htis stage there are no nonlinearities anymore
        self.encoder_flat = tf.contrib.layers.flatten(self.conv_block_5)
        self.z_mean = tf.layers.dense(self.encoder_flat, hidden_size)
        self.z_log_sigma_sq = tf.layers.dense(self.encoder_flat, hidden_size)

        # Sampling the bottleneck given its mean and variance
        self.eps = tf.random_normal(tf.shape(self.z_mean), 0, 1)
        self.z = self.z_mean + self.eps * tf.sqrt(tf.exp(self.z_log_sigma_sq))


    def __decoder(self, input_shape):
        """Setting up the decoding part of autoencoder"""
        # Reversing encoder part until convolution blocks
        # Using the shapes of corresponding encoder blocks to get at post convolution layer
        self.decoder_flat = tf.layers.dense(self.z, self.encoder_flat.shape[1], activation=tf.nn.relu)
        self.deconv_block_5 = tf.reshape(self.decoder_flat, tf.shape(self.conv_block_5))

        # Applying deconvolution layers
        # Final restored frame is also deconvolution output
        # So one must be careful when changing __undo_conv_block (especially activations)
        self.deconv_block_4 = VAEAutoencoder.__undo_conv_block(self.deconv_block_5, 256, 256, self.dropout_share)
        self.deconv_block_3 = VAEAutoencoder.__undo_conv_block(self.deconv_block_4, 128, 128, self.dropout_share)
        self.deconv_block_2 = VAEAutoencoder.__undo_conv_block(self.deconv_block_3, 64, 64, self.dropout_share)
        self.deconv_block_1 = VAEAutoencoder.__undo_conv_block(self.deconv_block_2, 32, 32, self.dropout_share)
        self.restored_frame = VAEAutoencoder.__undo_conv_block(self.deconv_block_1, 16, 3, self.dropout_share)


    def __loss(self):
        """Defining loss used in training"""
        self.l2_loss = 0.5 * tf.reduce_mean(tf.pow(self.input_frame - self.restored_frame, 2))
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq  
                                                - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq))
        self.loss = tf.minimum(self.l2_loss + self.latent_loss_weight * self.latent_loss, 10**8)
 

    def __optimizer(self):
        """Defining the optimizatin routine"""
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def __init__(self, hidden_size, input_shape):
        """Combining everything together and initializing PCA autoencoder"""
        self.__define_placeholders(input_shape)
        self.__encoder(hidden_size)
        self.__decoder(input_shape)
        self.__loss()
        self.__optimizer()


    def compress(self, session, frames):
        """Return an intermediary hidden layer"""
        return session.run(self.z, feed_dict={self.input_frame: frames})


    def restore(self, session, frames):
        """Return frames pushed through autoencoder"""
        return session.run(self.restored_frame, feed_dict={self.input_frame: frames})


    def show_loss(self, session, frames, dropout_share=0.0, latent_loss_weight=0.0):
        """Return reconstruction loss for given frames"""
        return session.run(self.loss, feed_dict={self.input_frame: frames,
                                                 self.dropout_share: dropout_share,
                                                 self.latent_loss_weight: latent_loss_weight})


    def train_op(self, session, frames, lr, dropout_share=0.0, latent_loss_weight=0.0):
        """Perform one gradient step"""
        session.run(self.optimizer, feed_dict={self.input_frame: frames, 
                                               self.learning_rate: lr,
                                               self.dropout_share: dropout_share,
                                               self.latent_loss_weight: latent_loss_weight})

