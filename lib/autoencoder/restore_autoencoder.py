# The test file to check the validity of saved autoencoder model
# The actual encoding of videos is supposed to be done in different files
import tensorflow as tf
from tf_saver import TfSaver
import numpy as np
from autoencoder import FrameAutoencoder
from video_frame_generator import batch_frame_generator


frame_generator = batch_frame_generator(batch_size=32)

with tf.Session() as sess:
    encoder_model = FrameAutoencoder(session=sess, learning_rate=0.1)
    saver = TfSaver('../../data/models/autoencoder_folder')
    saver.load_latest_checkpoint(sess)
    print encoder_model.loss(next(frame_generator))
