ARG distro=noetic
ARG workspace_name="drake_ws"

FROM ros:${distro}-robot-buster as builder
ARG distro
SHELL ["/bin/bash", "-c"]

RUN apt-get update \ 
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends python3 python3-pip

FROM builder as devcontainer
ARG distro
ARG workspace_name
ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN mkdir /home/$USERNAME/$workspace_name
RUN mkdir /home/$USERNAME/$workspace_name/src

WORKDIR /home/$USERNAME/$workspace_name

RUN echo ". /ros_entrypoint.sh" >> ~/.bashrc
