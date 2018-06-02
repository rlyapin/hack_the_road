
# The master script to train an autoencoder model

# Adding lib directory to path to be able to import from neughbouring directories
import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import tqdm
from lib.autoencoder.vae_autoencoder import VAEAutoencoder
from lib.utils.comma_ai_generator import batch_frame_generator
from lib.utils.tf_saver import TfSaver


# Training autoencoder on comma ai video feed
tf.reset_default_graph()
frame_generator = batch_frame_generator("../data/comma/train/camera/", batch_size=8, skip_frames=10)
model = VAEAutoencoder(hidden_size=100, input_shape=[160, 320, 3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Loading most recent progress if present
    saver = TfSaver('../data/models/comma_vae_autoencoder_folder/')
    saver.load_latest_checkpoint(sess)

    print "Loss before training: "
    print model.show_loss(sess, next(frame_generator))

    print "Started training: "
    for i in tqdm.tqdm(range(10000)):
        model.train_op(sess, next(frame_generator), lr=1e-4, dropout_share=0.25, latent_loss_weight=0.0)

        if i % 100 == 0:
            saver.save_checkpoint(sess)

    print "Loss after training:"
    print model.show_loss(sess, next(frame_generator))