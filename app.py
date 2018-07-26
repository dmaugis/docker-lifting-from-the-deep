#!/usr/bin/env python
# -*- coding: utf-8 -*-


import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import sys
import cv2
import zmq
import argparse
import numpy as np
import zmqnparray as zmqa
import simplejson as json
import matplotlib.pyplot as plt
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
#IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image.png'
#IMAGE_FILE_PATH = '/images/' + sys.argv[1]
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

# parse arguments
parser = argparse.ArgumentParser(description='Estimates 2d and 3d human pose from photo.')
parser.add_argument('--display', action="store_true", default=False,help='display graphical result')
parser.add_argument('--zmq', action="store", default="tcp://*:5555", help='publish subscribe url')
args=parser.parse_args()

print args

def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    fig=plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        fig=plot_pose(single_3D)
        pass
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data 


# create zmq server
port = "5555"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

while True:
    #  Wait for next request from client
    image, extra = zmqa.recv(socket)
    if extra is not None and 'fname' in extra:
        print("Received request %s" % str(extra['fname']))
    else:
        print("Received request %s" % str(extra))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb
    orig_img_size = np.array(image.shape)
    print "shape: ", image.shape
    INPUT_SIZE = 368
    scale=1.0 
    if orig_img_size[0] > orig_img_size[1]:
        scale = INPUT_SIZE / (orig_img_size[0] * 1.0)
    else:
        scale = INPUT_SIZE / (orig_img_size[1] * 1.0)
    image_size = np.round(orig_img_size * scale).astype(np.int32)
    image = cv2.resize(image, (0, 0), fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
 
    """
    INPUT_SIZE = 368

    self.orig_img_size = np.array(image_size)
    self.scale = utils.config.INPUT_SIZE / (self.orig_img_size[0] * 1.0)
    self.img_size = np.round(self.orig_img_size * self.scale).astype(np.int32)

    image = cv2.resize(image, (0, 0), fx=self.scale,fy=self.scale,interpolation=cv2.INTER_CUBIC)

    """
    try:
    #if True:
        # create pose estimator
        pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

        # load model
        pose_estimator.initialise()

        # estimation
        pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

        # close model
        pose_estimator.close()

        # Show 2D and 3D poses
        pltimg=display_results(image, pose_2d, visibility, pose_3d)

        print "pose_2d", pose_2d

        # print json
        lift={}
        lift["pose2d"]=pose_2d.round(2).tolist()
        lift["pose3d"]=pose_3d.round(2).tolist()
        lift["visibility"]=visibility.tolist()
        print json.dumps(lift)
        zmqa.send(socket,pltimg,extra=lift)
    except:
    #else:
        print "Unexpected error:", sys.exc_info()[0]
        lift={}
        zmqa.send(socket,image,extra=lift)





