# The master script to train lstm model for parking mode prediction

import tensorflow as tf
import numpy as np
import cv2
from tf_saver import TfSaver
from lstm_model import LSTMClassifier
from sequence_generator import batch_sequence_generator

sequence_generator = batch_sequence_generator(batch_size=8)

tf.reset_default_graph()
with tf.Session() as sess:
    lstm_model = LSTMClassifier(session=sess, learning_rate=1)
    saver = TfSaver('../../data/models/lstm_parking_predictor')

    saver.load_latest_checkpoint(sess)

    print "Loss before training"
    print lstm_model.loss(*next(sequence_generator))

    print "Started training"
    for i in range(1000):
        print i
    	# train_op method requires an input tensor of [None, 10] shape
        lstm_model.train_op(*next(sequence_generator))
        if i % 100 == 0:
            print "Current loss: "
            print lstm_model.loss(*next(sequence_generator))
            saver.save_checkpoint(sess)

    print "Loss after training:"
    print lstm_model.loss(*next(sequence_generator))