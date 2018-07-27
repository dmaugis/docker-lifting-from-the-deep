#!/usr/bin/env python
# -*- coding: utf-8 -*-

help = """reqimages
 
Usage:
  reqfiles.py <files>...
 
Options:
  -h --help          This help.
 
(c) Sample Copyright
"""

import sys
import cv2
import zmq
import numpy as np
import zmqnparray as zmqa
import simplejson as json
import math
import matplotlib.pyplot as plt
from docopt import docopt
import os
import os.path

arguments = docopt(help)
#print(arguments)
def draw_limbs(image, pose_2d, visible):
    JOINT_DRAW_SIZE = 3
    LIMB_DRAW_SIZE = 2
    """Draw the 2D pose without the occluded/not visible joints."""
    NORMALISATION_COEFFICIENT = 1280*720
    _COLORS = [
        [0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0],
        [170, 255, 0], [255, 170, 0], [255, 0, 0], [255, 0, 170],
        [170, 0, 255]
    ]
    _LIMBS = np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9,
                       9, 10, 11, 12, 12, 13]).reshape((-1, 2))

    _NORMALISATION_FACTOR = int(math.floor(math.sqrt(368 * 368 / NORMALISATION_COEFFICIENT)))
    #print("LIMB_DRAW_SIZE*_NORMALISATION_FACTOR", LIMB_DRAW_SIZE, " ",_NORMALISATION_FACTOR, " ",LIMB_DRAW_SIZE*_NORMALISATION_FACTOR,LIMB_DRAW_SIZE*_NORMALISATION_FACTOR)
    for oid in range(pose_2d.shape[0]):
        for lid, (p0, p1) in enumerate(_LIMBS):
            if not (visible[oid][p0] and visible[oid][p1]):
                continue
            y0, x0 = pose_2d[oid][p0]
            y1, x1 = pose_2d[oid][p1]
            cv2.circle(image, (x0, y0), JOINT_DRAW_SIZE *_NORMALISATION_FACTOR +1, _COLORS[lid], -1)
            cv2.circle(image, (x1, y1), JOINT_DRAW_SIZE*_NORMALISATION_FACTOR +1, _COLORS[lid], -1)
            cv2.line(image, (x0, y0), (x1, y1),_COLORS[lid], LIMB_DRAW_SIZE*_NORMALISATION_FACTOR +1, 16)


def plot_pose(pose):
    """Plot the 3D pose showing the joint connections."""
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]

    def joint_color(j):
        """
        TODO: 'j' shadows name 'j' from outer scope
        """

        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in range(1, 4):
            _c = 1
        if j in range(4, 7):
            _c = 2
        if j in range(9, 11):
            _c = 3
        if j in range(11, 14):
            _c = 4
        if j in range(14, 17):
            _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)
    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                   c=col, marker='o', edgecolor=col)
    smallest = pose.min()
    largest = pose.max()
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    return fig

def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    fig=plt.figure()
    data1=None
    data2=None
    if data_2d is not None:
       draw_limbs(in_image, data_2d, joint_visibility)
       plt.imshow(in_image)
       plt.axis('off')
       fig.canvas.draw()
       data1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
       data1 = data1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if data_3d is not None:
       for single_3D in data_3d:
           # or plot_pose(Prob3dPose.centre_all(single_3D))
           fig=plot_pose(single_3D)
           pass
       fig.canvas.draw()
       data2 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
       data2 = data2.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return data2,data1 





file_list=arguments.pop("<files>", None)

context = zmq.Context()

#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

for fname in file_list:
    if os.path.isfile(fname) and os.access(fname, os.R_OK):
        A=cv2.imread(fname,1)
        if A is None:
            print("[%s] Could not read image" % (fname))
        else:
            arguments['fname']=fname
            #print("[%s] Sending request… " % (fname) )
            print fname 
            zmqa.send(socket,A,extra=arguments)
            #zmqa.send(socket,A)
            #  Get the reply.
            B,extra= zmqa.recv(socket)
            #print("[%s] Received reply %s" % (fname,str(extra)))
            cv2.imshow('req',A)
            if B is not None:
               cv2.imshow('rep',B)
            else:
               #try:
               pose2d=None
               pose3d=None
               visibility=None
               if 'pose2d' in extra:
                   #print "extra: ", extra
                   pose2d=np.array(extra['pose2d'],dtype='int16')
               if 'visibility' in extra:
                   visibility=np.array(extra['visibility'],dtype='bool')
               if 'pose3d' in extra:
                   pose3d=np.array(extra['pose3d'],dtype='float64')
               #print "pose2d: ", pose2d
               #print "pose3d: ", pose3d
               img3d,img2d=display_results(A, pose2d, visibility, pose3d)
               if img3d is not None: cv2.imshow("pose3d",img3d)
               if img2d is not None: cv2.imshow("pose2d",img2d)
               #except:
               else:
                   pass
            cv2.waitKey(10)
    else:
        print("[%s] could not access file" % (fname))


cv2.destroyAllWindows()
