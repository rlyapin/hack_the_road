# The function needed to normalize images from video feed and restore them

import cv2
import numpy as np

# IMAGENET_MEANS = [103.939, 116.779, 123.68]
IMAGENET_MEANS = [0.40760392, 0.45795686, 0.48501961]
IMAGE_SHAPE = [224, 224, 3]

def process_image(img):
    processed_image = cv2.resize(img, tuple(IMAGE_SHAPE[:2]))
    processed_image = processed_image.astype(float)
    processed_image /= 255
    for x in range(3):
        processed_image[:, :, x] -= IMAGENET_MEANS[x]
    return processed_image
        
def restore_image(img):
    restored_image = np.array(img)
    for x in range(3):
        restored_image[:, :, x] += IMAGENET_MEANS[x]
    restored_image.clip(0, 1)
    restored_image *= 255
    return restored_image.astype(np.uint8)