 
# The code that creates frame generators on top of Comma AI dataset

# One of shortcomings of original frame generators from .mp4 is speed
# Generators include costly cv2.resize() calls and fill batch generators 1-by-1
# This resulted in generator calls taking longer than training calls
# For that reason comma_ai generator would only work with batches and do little preprocessing

import glob
import h5py
import numpy as np

def batch_frame_generator(comma_dir, batch_size, skip_frames=0):
    """Generator to output frames from Comma AI dataset

    Comma AI dataset comprises of several .h5 files
    This files are wrappers on top of np.ndarray with [None, 3, 160, 320] shape and uint8 type
    Hence before yielding a batch I move to move channel in the back and normalize tensors

    Args:
        comma_dir: a folder with Comma AI .h5 files (it is assumed I can split them between train and test)
        batch_size: a numebr of frames in a single batch
        skip_frames: the number of frames discarded until yielding a frame (to drop similar frames and interate faster)
    Returns:
        a batch of random frames with [None, 160, 320, 3] shape
    """ 

    # Fetching all Comma AI files
    comma_files = glob.glob(comma_dir + "*")

    while True:
        for h_file in comma_files:
            
            # Following the notation in Comma AI dataset, X is the collection of all frames from a single video
            # X is supposed to have [None, 3, 160, 320] shape and uint8 type
            X = h5py.File(h_file, 'r')["X"]
            n_frames = X.shape[0]
            n_batches = int(n_frames / (batch_size * (1 + skip_frames)))

            # Yielding normalized frames in consecutive batches to speed things up
            # 100x speedup compared to yielding scattered indices
            # Note that going for skip_frames > 0 considerably slows everything down
            for i in range(n_batches):
                picked_frames = X[(1 + skip_frames) * i * batch_size: 
                                  (1 + skip_frames) * (i + 1) * batch_size: 
                                  (1 + skip_frames)]
                yield picked_frames.transpose([0, 2, 3, 1]) / 255.0
        np.random.shuffle(comma_files)

