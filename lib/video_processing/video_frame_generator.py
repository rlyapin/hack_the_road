# The code that takes creates the generator that produces all .mp4 files in data/video

import os
import cv2
import glob
import random
import numpy as np

IMAGE_SHAPE = [224, 224, 3]

def single_frame_generator():
    while True:
        for file in glob.glob("../../data/video/*.mp4"):
            video_reader = cv2.VideoCapture(file)
            # Lame syntax on OpenCV side: 7th parameter is the number of frames
            n_frames = int(video_reader.get(7))
            # 1 second in OpenCV VideoCapture method would take 25 frames (?)
            # Im fine sampling frames every 0.2 seconds
            sampled_indices = range(0, n_frames, 5)
            for index in sampled_indices:
                # Lame syntax on OpenCV side: 1th parameter is pointer location in the video
                video_reader.set(1, index)
                bool_flag, frame = video_reader.read()
                if bool_flag == True:
                    frame = cv2.resize(frame, tuple(IMAGE_SHAPE[:2]))
                    yield frame
        
def batch_frame_generator(batch_size):
    frame_generator = single_frame_generator()
    generated_frames = np.zeros([batch_size] + IMAGE_SHAPE)
    while True:
        for i in range(batch_size):
            generated_frames[i, :, :, :] = next(frame_generator)
        yield generated_frames

def random_frame_generator(video_dir):
    all_files = glob.glob(video_dir + "*.mp4")
    while True:
        picked_file = random.choice(all_files)
        video_reader = cv2.VideoCapture(picked_file)
        # Lame syntax on OpenCV side: 7th parameter is the number of frames
        n_frames = int(video_reader.get(7))
        picked_frame = random.choice(range(n_frames))
        # Lame syntax on OpenCV side: 1th parameter is pointer location in the video
        video_reader.set(1, picked_frame)
        _, random_frame = video_reader.read()  
        yield cv2.resize(random_frame, tuple(IMAGE_SHAPE[:2]))      
