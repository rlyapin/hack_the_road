# The test file to check the validity of saved autoencoder model
# The actual encoding of videos is supposed to be done in different files
import tensorflow as tf
from tf_saver import TfSaver
import numpy as np
from autoencoder import FrameAutoencoder


test_data = np.arange(10).reshape(1, 10)

with tf.Session() as sess:
    encoder_model = FrameAutoencoder(session=sess, learning_rate=0.1)
    saver = TfSaver('../../data/models/autoencoder_folder')
    saver.load_latest_checkpoint(sess)
    print encoder_model.loss(test_data)
