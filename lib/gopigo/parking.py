#!/usr/bin/env python

from gopigo import *
import argparse
import os
import picamera
import sys

parser = argparse.ArgumentParser("parking")
parser.add_argument("type", help="Type of movement", type=str)
parser.add_argument("run", help="Run identifier", type=int)
parser.add_argument("--camera", help="Use camera or not", type=int)
args = parser.parse_args()

print('Type {}, run {}'.format(args.type, args.run))

def go():
    commands = []
    if args.type == 'parallel_backward':
        commands = [
            (lambda: fwd()  , 3  , 'led_off'),
            (lambda: bwd()  , 0.5, 'led_on'),
            (lambda: left() , 0.7, 'led_on'),
            (lambda: bwd()  , 1  , 'led_on'),
            (lambda: right(), 1  , 'led_on'),
            (lambda: bwd()  , 0.5, 'led_on'),
        ]
    elif args.type == 'move_forward':
        commands = [
            (lambda: fwd()  , 5  , 'led_off'),
        ]
    elif args.type == 'parallel_forward':
        commands = [
            (lambda: fwd()  , 1  , 'led_off'),
            (lambda: right(), 0.01, 'led_on'),
            (lambda: fwd()  , 0.7, 'led_on'),
            (lambda: left() , 0.01, 'led_on'),
            (lambda: fwd()  , 0.5, 'led_on'),
        ]
    else:
        print('Type not supported. Supported types: parallel_backward, parallel_forward, move_forward')
        return

    RIGHT_LED = 0

    camera = None
    if args.camera == 1:
        camera = picamera.PiCamera()
        camera.vflip = True
        camera.resolution = (1280, 720)
        camera.framerate = (60)
        camera.start_recording('videos/{}_{}.h264'.format(args.type, args.run))

    for cmd, sleep, led in commands:
        cmd()
        for i in range(int(sleep / 0.01)):
            if led == 'led_on' and i % 5 == 0:
                if i % 2 == 0:
                    led_on(RIGHT_LED)
                else:
                    led_off(RIGHT_LED)

            time.sleep(0.01)

    if args.camera == 1:
        camera.stop_recording()

    led_off(RIGHT_LED)
    stop()

go()