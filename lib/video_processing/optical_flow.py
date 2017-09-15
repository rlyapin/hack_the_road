# The function wrapper on top of optical flow

import cv2 
import numpy as np

def optical_flow(frame1, frame2):
    frame1_map = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_map = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(frame1_map, frame2_map, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    return np.ma.masked_invalid(mag).mean(), np.ma.masked_invalid(ang).mean()
