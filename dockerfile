ARG distro=noetic
ARG workspace_dir=~/drake_ws

FROM ros:${distro}-robot-buster AS builder
ARG workspace_dir
ARG distro
SHELL ["/bin/bash", "-c"]
WORKDIR ${workspace_dir}

ADD . ./src/drake

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    . /opt/ros/${distro}/setup.bash && \
    rosdep install --from-paths src --ignore-src -y && \
    catkin_make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

