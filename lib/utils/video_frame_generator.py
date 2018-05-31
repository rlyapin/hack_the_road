
# The code that creates various generators to train vae on .mp4 files in some location

import cv2
import glob
import random
import numpy as np
from image_processing import process_image

IMAGE_SHAPE = [224, 224, 3]

# Handling lame OpenCV methods / syntax when parameters are passed as integer
CV2_N_FRAMES_CODE = 7
CV2_VIDEO_POINTER = 1


def fixed_frame_generator(video_dir):
    """Generator that outputs a single fixed frame

    The ultimate purpose of this generator is debugging
    That is why the returned frame would be same 

    Args:
        video_dir: a folder with .mp4 files that
    Returns:
        a fixed frame
    """

    all_files = glob.glob(video_dir + "*.mp4")
    all_files.sort()
    while True:
        video_reader = cv2.VideoCapture(all_files[0])

        # Picking a frame in the middle of the video
        n_frames = int(video_reader.get(CV2_N_FRAMES_CODE))
        index = n_frames / 2
        video_reader.set(CV2_VIDEO_POINTER, index)
        _, frame = video_reader.read()
        yield process_image(frame)


def random_frame_generator(video_dir):
    """Generator that outputs a single random frame

    This generator should provide the best shuffling of training data
    Happens as each frame is sampled anew independent from previous samples 

    Args:
        video_dir: a folder with .mp4 files that
    Returns:
        a random frame
    """

    all_files = glob.glob(video_dir + "*.mp4")
    while True:
        picked_file = random.choice(all_files)

        video_reader = cv2.VideoCapture(picked_file)
        n_frames = int(video_reader.get(CV2_N_FRAMES_CODE))
        picked_frame = np.random.choice(n_frames)
        video_reader.set(CV2_VIDEO_POINTER, picked_frame)
        _, random_frame = video_reader.read()  
        yield process_image(random_frame)  


def single_frame_generator(video_dir):
    """A generator that outputs all frames within directory one at a time

    It should provide worse data shuffling than random_frame_generator
    However, it compensates for that with faster data generation
    To have some shuffling it still changes the order of files fed into generator

    Args:
        video_dir: a folder with .mp4 files that
    Returns:
        a video frame
    """

    all_files = glob.glob(video_dir + "*.mp4")
    while True:
        for file in all_files:
            video_reader = cv2.VideoCapture(file)

            n_frames = int(video_reader.get(CV2_N_FRAMES_CODE))
            # 1 second in OpenCV VideoCapture method would take 25 frames (?)
            # Im fine sampling frames every 0.2 seconds
            sampled_indices = range(0, n_frames, 5)

            for index in sampled_indices:
                video_reader.set(CV2_VIDEO_POINTER, index)
                bool_flag, frame = video_reader.read()
                if bool_flag == True:
                    yield process_image(frame)

        random.shuffle(all_files)

        
def batch_frame_generator(generator, batch_size):
    """A generator that outputs a batch of frames

    To do that it iteratively makes calls to a single frame generator

    Args:
        generator: a generator that yields single frame in IMAGE_SHAPE format
        batch_size: the size of geenrated batches
    Returns:
        a batch of video frames
    """

    generated_frames = np.zeros([batch_size] + IMAGE_SHAPE)
    while True:
        for i in range(batch_size):
            generated_frames[i, :, :, :] = next(generator)
        yield generated_frames
 
