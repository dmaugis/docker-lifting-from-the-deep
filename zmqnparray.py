
import zmq
import numpy as np
import cv2


def send(socket, array, flags=0, copy=True, track=False,extra=None):
    A = np.ascontiguousarray(array)
    md = {}
    if extra is not None:
       md["extra"]=extra
    md['dtype'] = str(A.dtype)
    md['shape'] = A.shape
    #cv2.imshow('zmqnparray::send',A)
    #cv2.waitKey(0)
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv(socket, flags=0, copy=True, track=False):
    extra=None
    md = socket.recv_json(flags=flags)
    #print "md: ",md
    msg = socket.recv(flags=flags, copy=copy, track=track)
    extra=md.pop('extra', None)
    A=None
    if 'dtype' in md:
       if 'shape' in md:
          A = np.frombuffer(msg, dtype=md['dtype'])
          A = A.reshape(md['shape'])
          #cv2.imshow('zmqnparray::recv',A)
          #cv2.waitKey(0)
    return A, extra

