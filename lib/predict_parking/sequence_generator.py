# The code that takes creates the generator that outputs compressed info about car actions
# It is assumed that each seqeunce is stored in a relatively small file
# For consistency I have a specified MAX_SEQUENCE_LENGTH
# If the compressed data has less entries it is padded with extra zero values
# If compressed data has more entries I select only the last one
# Note that I also need to output labels

import os
import cv2
import glob
import numpy as np
import random

MAX_SEQUENCE_LENGTH = 50
COMPRESSION_SIZE = 51

def single_sequence_generator():
    while True:
        # Shuffling files to train on (to add randomness to the labels)
        all_files = glob.glob("../../data/processed_video/*.txt")
        random.shuffle(all_files)

        for file in all_files:
            compressed_data = np.loadtxt(file)
            if compressed_data.shape[0] < MAX_SEQUENCE_LENGTH:
                # Adding zero padding from above if the sample is too small
                compressed_data = np.lib.pad(compressed_data, 
                                             ((MAX_SEQUENCE_LENGTH - compressed_data.shape[0], 0), (0, 0)),
                                             'constant', constant_values=(0,))

            if compressed_data.shape[0] > MAX_SEQUENCE_LENGTH:
                # Leaving last observations if the number os too big
                compressed_data = compressed_data[-MAX_SEQUENCE_LENGTH:, :]

            # Returnng label as (1, 1) numpy array
            label = np.array([int("parking" in file)]).reshape(1, 1)

            yield compressed_data, label
        
def batch_sequence_generator(batch_size):
    frame_generator = single_sequence_generator()
    generated_sequences = np.zeros((batch_size, MAX_SEQUENCE_LENGTH, COMPRESSION_SIZE))
    generated_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            seqeunce, label = next(frame_generator)
            generated_sequences[i, :, :] = seqeunce
            generated_labels[i, :] = label
        yield generated_sequences, generated_labels
