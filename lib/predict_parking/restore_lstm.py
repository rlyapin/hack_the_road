# The test file to check the validity of saved lstm model
import tensorflow as tf
from tf_saver import TfSaver
import numpy as np
from lstm_model import LSTMClassifier
from sequence_generator import batch_sequence_generator


sequence_generator = batch_sequence_generator(batch_size=8)

with tf.Session() as sess:
    lstm_model = LSTMClassifier(session=sess, learning_rate=1)
    saver = TfSaver('../../data/models/lstm_parking_predictor')
    saver.load_latest_checkpoint(sess)
    print lstm_model.loss(*next(sequence_generator))
