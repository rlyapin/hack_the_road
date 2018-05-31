
# The code to implement a PCA autoencoder
# By PCA autoencoder I mean a NN with one bottleneck hidden layer without any nonlinearities
# The code is simple enough yet I still break it down in lots of smaller functions to prepare for VAE

import tensorflow as tf


class PCAAutoencoder:
    def __define_placeholders(self, input_shape):
        """Setting up autoencoder variables"""

        # None in shape stands for the batch size
        self.input_frame = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])


    def __encoder(self, hidden_size):
        """Setting up the encoding part of autoencoder"""

        # For this specific use case with PCA the encoder is a simple dense layer
        # Here I am additionally following a notation from VAE and call my hidden layer "z"
        # Note that no nonlinearity is applied
        self.z = tf.layers.dense(tf.contrib.layers.flatten(self.input_frame), hidden_size)


    def __decoder(self, input_shape):
        """Setting up the decoding part of autoencoder"""

        # For this specific use case with PCA the decoder is also a simple dense layer
        output_size = reduce(lambda x,y: x * y, input_shape)
        self.restored_frame = tf.reshape(tf.layers.dense(self.z, output_size), tf.shape(self.input_frame))


    def __loss(self):
        """Defining loss used in training"""

        # There is no variational component so I am simply going for L2 loss
        self.loss = 0.5 * tf.reduce_mean(tf.pow(self.input_frame - self.restored_frame, 2))


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


    def show_loss(self, session, frames):
        """Return reconstruction loss for given frames"""
        return session.run(self.loss, feed_dict={self.input_frame: frames})


    def train_op(self, session, frames, lr):
        """Perform one gradient step"""
        session.run(self.optimizer, feed_dict={self.input_frame: frames, self.learning_rate: lr})

