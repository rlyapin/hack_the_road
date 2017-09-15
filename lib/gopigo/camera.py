#!/usr/bin/env python

from gopigo import *
import argparse
import os
import picamera
import sys

parser = argparse.ArgumentParser("parking")
parser.add_argument("--run", help="Run identifier", type=int)
parser.add_argument("--type", help="Type", type=str)
parser.add_argument("--seconds", help="Seconds", type=int)
args = parser.parse_args()

print('Run {}, type {}, {} seconds:'.format(args.run, args.type, args.seconds))

def go():
    camera = picamera.PiCamera()
    camera.vflip = True
    camera.resolution = (1280, 720)
    camera.framerate = 25
    camera.start_recording('videos/{}_run_{}.h264'.format(args.type, args.run))

    time.sleep(args.seconds)
    camera.stop_recording()

go()