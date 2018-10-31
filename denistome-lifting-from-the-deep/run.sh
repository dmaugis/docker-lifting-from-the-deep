xhost +
nvidia-docker run -it --network host --rm -e "DISPLAY=unix:0.0"  -v /tmp/.X11-unix:/tmp/.X11-unix:rw --privileged  -v $(realpath ./shared):/shared  denistome/lifting-from-the-deep
