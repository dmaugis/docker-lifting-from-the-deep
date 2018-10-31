FROM denistome/lifting-from-the-deep

RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
                      git wget python-opencv python-tk 

RUN pip install simplejson
RUN pip install pyzmq
RUN pip install docopt

# Set the working directory to /Lifting-from-the-Deep
WORKDIR /Lifting-from-the-Deep
ADD zmqnparray.py applications/
ADD rep.py /Lifting-from-the-Deep/applications 
CMD python applications/rep.py









