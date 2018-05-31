# Wrapper around tensorflow saver to allow for saving / restoring models in different files
# Thanks to omikad for implementation

import os
import tensorflow as tf

class TfSaver:
    def __init__(self, logdir):
        self.checkpoint_dir = os.path.join(logdir, "checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.saver = tf.train.Saver()
        
    def load_latest_checkpoint(self, session):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(session, latest_checkpoint)
            
    def save_checkpoint(self, session):
        self.saver.save(session, self.checkpoint_path)
