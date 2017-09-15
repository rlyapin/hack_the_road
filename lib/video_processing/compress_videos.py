# The function first that takes all .mp4 videos in data/video folder
# Next it collects the pretrained frame autoencoder model and compresses video frames
# Additionally, optical flow information is added to the embedding

import numpy as np
import cv2
import tensorflow as tf
import glob
from tf_saver import TfSaver
from autoencoder import FrameAutoencoder
from optical_flow import optical_flow

# Currently compression size is: 49 entries for autoencoder and 2 entries for optical flow
IMAGE_SHAPE = [224, 224, 3]
COMPRESSION_SIZE = 51

with tf.Session() as sess:
    # Loading trained autoencoder
    encoder_model = FrameAutoencoder(session=sess, learning_rate=0.1)
    saver = TfSaver('../../data/models/autoencoder_folder')
    saver.load_latest_checkpoint(sess)

    # Compressing each .mp4 file in video folder
    for file in glob.glob("../../data/video/*.mp4"):
        video_reader = cv2.VideoCapture(file)
        # Lame syntax on OpenCV side: 7th parameter is the number of frames
        n_frames = int(video_reader.get(7))
        # 1 second in OpenCV VideoCapture method would take 25 frames (?)
        # Im fine sampling frames every 0.2 seconds
        sampled_indices = range(0, n_frames - 1, 500)
        compressed_file = np.zeros((len(sampled_indices) - 1, COMPRESSION_SIZE))

        # Lame syntax on OpenCV side: 1th parameter is pointer location in the video
        # Grabbing the first frame
        video_reader.set(1, sampled_indices[0])
        _, current_frame = video_reader.read()
        current_frame = cv2.resize(current_frame, tuple(IMAGE_SHAPE[:2]))

        print compressed_file.shape
        for i in range(1, len(sampled_indices)):
            # Updated the current frame
            previous_frame = current_frame
            video_reader.set(1, sampled_indices[i])
            _, current_frame = video_reader.read()
            current_frame = cv2.resize(current_frame, tuple(IMAGE_SHAPE[:2]))

            # Getting frame embedding from autoencoder
            compressed_file[i - 1, :COMPRESSION_SIZE-2] = encoder_model.compress(np.expand_dims(current_frame, axis=0))

            # Getting information about optical flow
            mag, ang = optical_flow(previous_frame, current_frame)
            compressed_file[i - 1, COMPRESSION_SIZE-2] = mag
            compressed_file[i - 1, COMPRESSION_SIZE-1] = ang

        # Determining output directory
        # Using the file that original filename ends in "/file.mp4"
        filename = file.split("/")[-1][:-4]
        output_path = "../../data/processed_video/" + filename + ".txt"
        np.savetxt(output_path, compressed_file)

