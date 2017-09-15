# The master script to train autoencoder model

import tensorflow as tf
import numpy as np
import cv2
from tf_saver import TfSaver
from autoencoder import FrameAutoencoder
from video_frame_generator import batch_frame_generator

frame_generator = batch_frame_generator(batch_size=32)

tf.reset_default_graph()
with tf.Session() as sess:
    encoder_model = FrameAutoencoder(session=sess, learning_rate=0.1)
    saver = TfSaver('../../data/models/autoencoder_folder')

    saver.load_latest_checkpoint(sess)

    print "Loss before training"
    print encoder_model.loss(next(frame_generator))

    print "Started training"
    for i in range(1000):
        print i
    	# train_op method requires an input tensor of [None, 10] shape
        encoder_model.train_op(next(frame_generator))
        if i % 100 == 0:
            print "Current loss: "
            print encoder_model.loss(next(frame_generator))
            saver.save_checkpoint(sess)

    print "Loss after training:"
    print encoder_model.loss(next(frame_generator))