#!/usr/bin/env python
# -*- coding: utf-8 -*-

help = """zmqlift
 
Usage:
  zmqlift
  zmqlift --display
  zmqlift --zmq <string>
  zmqlift -h | --help

Options:
  --display          display received.
  -h --help          This help.
 
(c) Sample Copyright
"""

import sys
import cv2
import zmq
import argparse
import numpy as np
import zmqnparray as zmqa
import simplejson as json
import matplotlib.pyplot as plt
from docopt import docopt
import os
import os.path
from os.path import dirname, realpath


import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose


DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
#IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image.png'
#IMAGE_FILE_PATH = '/images/' + sys.argv[1]
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

# parse arguments
#parser = argparse.ArgumentParser(description='Estimates 2d and 3d human pose from photo.')
#parser.add_argument('--display', action="store_true", default=False,help='display graphical result')
#parser.add_argument('--zmq', action="store", default="tcp://*:5555", help='publish subscribe url')
#args=parser.parse_args()

args=docopt(help)
#print args


def size_image(image):
    orig_img_size = np.array(image.shape)
    scale = INPUT_SIZE / (orig_img_size[0] * 1.0)
    if orig_img_size[0] < orig_img_size[1]:
        scale = INPUT_SIZE / (orig_img_size[1] * 1.0)
    image_size = np.round(orig_img_size * scale).astype(np.int32)
    image = cv2.resize(image, (0, 0), fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

    delta_w = INPUT_SIZE - image_size[1]
    delta_h = INPUT_SIZE - image_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 
    image_size =np.array(image.shape)
    return image
    


# create zmq server
port = "5555"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

INPUT_SIZE = 368

image_size = np.array((INPUT_SIZE,INPUT_SIZE, 3)).astype(np.int32)
# create pose estimator
pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)
# load model
pose_estimator.initialise()
plt2d=None
plt3d=None

while True:
    plt2d=None
    plt3d=None
    #  Wait for next request from client
    image, extra = zmqa.recv(socket)
    fname=""
    if extra is not None and 'fname' in extra:
        fname=extra['fname']
        print("[%s]" % fname)
    else:
        print(".")
    #cv2.imshow('req',image)
    #cv2.waitKey(0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb
    image = size_image(image)
    plt2d=None
    plt3d=None
    try:
    #if True:
        pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
        if 'display' in args:
           plt3d,plt2d=display_results(image, pose_2d, visibility, pose_3d)

        lift={}
        lift["pose2d"]=pose_2d.round(2).tolist()
        lift["pose3d"]=pose_3d.round(2).tolist()
        lift["visibility"]=visibility.tolist()
        #cv2.imshow('req',plt3d)
        #cv2.waitKey(0)
        zmqa.send(socket,plt3d,extra=lift)
    except:
    #else:
        print "!@*!! error:", sys.exc_info()[0]
        lift={}
        lift['error']=str(sys.exc_info()[0])
        lift['fname']=fname
        zmqa.send(socket,None,extra=lift)

# close model
pose_estimator.close()




