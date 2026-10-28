FROM ubuntu:24.04

LABEL maintainer="Thomas Laebe <laebe@ipb.uni-bonn.de>"

# -----------------------------------------------------------------------------------------
# Needed Packages
# python3-venv: Ubuntu 24.04 pip is "externally managed" (PEP 668), so we install into a venv.
# libpng16-16t64 / libglfw3: needed by open3d; libgl1 + libglib2.0-0 etc.: needed by opencv.
# git, cmake, build-essential, python3-dev: build the apriltag library.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-pip python3-venv python3-dev build-essential cmake git ca-certificates \
        libc++1 libgomp1 libpng16-16t64 libglfw3 \
        libgl1 libegl1 libusb-1.0-0 libglib2.0-0t64 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip

# -----------------------------------------------------------------------------------------
# AprilTag library, v3.4.2 (the pip version is too old, point numbering differs).
# numpy must be installed before the build, otherwise the python wrapper is not built.
WORKDIR /root
RUN git clone --branch 'v3.4.2' https://github.com/AprilRobotics/apriltag
WORKDIR /root/apriltag
RUN pip install --no-cache-dir numpy
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/opt/venv/bin/python3 \
    && cmake --build build --target install

# Library path has to be set so that python finds the apriltag library
ENV LD_LIBRARY_PATH=/usr/local/lib

ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

# -----------------------------------------------------------------------------------------
# The actual program
RUN pip --no-cache-dir install ipdb

WORKDIR /root/calib
COPY scripts ./scripts
COPY src ./src
COPY pyproject.toml .
RUN pip --no-cache-dir install -e .

WORKDIR /root
COPY entrypoint_apriltag_extraction.bash /root/entrypoint_apriltag_extraction.bash

CMD /root/entrypoint_apriltag_extraction.bash
