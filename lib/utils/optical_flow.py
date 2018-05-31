# The function wrapper on top of optical flow

import cv2 
import numpy as np

def optical_flow(frame1, frame2):
    """A function to calculate the optical flow

    Details on what optical flow is are given here:
    https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html

    Args:
        frame1: a numpy.ndarray with [None, None, 3] shape
        frame2: a numpy.ndarray with the same shape
    Returns:
        average magnitude of movement between images
        average angle of movement between images
    """

    frame1_map = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_map = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(frame1_map, frame2_map, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    return np.ma.masked_invalid(mag).mean(), np.ma.masked_invalid(ang).mean()
