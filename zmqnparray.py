
import zmq
import numpy as np


def send(socket, array, flags=0, copy=True, track=False,extra=None):
    A = np.ascontiguousarray(array)
    md = dict(
         dtype = str(A.dtype),
         shape = A.shape,
    )
    if extra is not None:
       md["extra"]=extra
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv(socket, flags=0, copy=True, track=False):
    extra=None
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    extra=md.pop('extra', None)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape']),extra

