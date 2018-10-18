FROM nvidia/opengl:1.0-glvnd-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libglm-dev libx11-dev libegl1-mesa-dev \
	libpng-dev xorg-dev cmake libjpeg-dev python3-opencv \
	python3-dev build-essential pkg-config git curl wget automake libtool

RUN git clone https://github.com/glfw/glfw.git && cd glfw && mkdir build && cd build && cmake .. && make &&  make install
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py
