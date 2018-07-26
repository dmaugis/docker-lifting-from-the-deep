FROM tensorflow/tensorflow:1.7.1-devel-gpu

RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
                      git wget python-opencv python-tk 

RUN pip install simplejson
RUN pip install pyzmq

# Set the working directory to /Lifting-from-the-Deep
WORKDIR /
RUN git clone https://github.com/DenisTome/Lifting-from-the-Deep-release.git Lifting-from-the-Deep
WORKDIR /Lifting-from-the-Deep
RUN ./setup.sh
#RUN python applications/demo.py
ADD zmqnparray.py applications/
ADD app.py /Lifting-from-the-Deep/applications 
#CMD python applications/demo.py
CMD python applications/app.py









