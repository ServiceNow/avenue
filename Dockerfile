#FROM pytorch/pytorch
FROM nvidia/opengl:1.0-glvnd-devel-ubuntu18.04

ARG PYTHON_VERSION=3.6

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libglm-dev libx11-dev libegl1-mesa-dev \
	libpng-dev xorg-dev cmake libjpeg-dev python3-opencv \
	python3-dev build-essential pkg-config git curl wget automake libtool ca-certificates

RUN git clone https://github.com/glfw/glfw.git && cd glfw && mkdir build && cd build && cmake .. && make &&  make install
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Install pytorch
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
COPY . .

#RUN git submodule update --init
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

WORKDIR /workspace
RUN chmod -R a+w /workspace


RUN mkdir /tmp/Avenue

COPY . /tmp/Avenue/

RUN chmod -R 777 /tmp/Avenue
RUN cd /tmp/Avenue &&  pip3 install -e .
RUN mkdir /tmp/avenue_assets
ENV AVENUE_ASSETS=/tmp/avenue_assets
RUN chmod -R 777 /tmp/avenue_assets

RUN pip install comet_ml

RUN pip install pyro-ppl
RUN pip install imageio
