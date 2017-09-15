# The master script to train autoencoder model

import tensorflow as tf
import numpy as np
import cv2
from tf_saver import TfSaver
from autoencoder import FrameAutoencoder


# Currently we use placeholder instead of real video frames
test_data = np.arange(10).reshape(1, 10)

tf.reset_default_graph()
with tf.Session() as sess:
    encoder_model = FrameAutoencoder(session=sess, learning_rate=0.1)
    saver = TfSaver('../../data/models/autoencoder_folder')

    print "Loss before training"
    print encoder_model.loss(test_data)

    print "Started training"
    for i in range(100):
    	# train_op method requires an input tensor of [None, 10] shape
        encoder_model.train_op(test_data)
        if i % 10 == 0:
            saver.save_checkpoint(sess)

    print "Loss after training:"
    print encoder_model.loss(test_data)