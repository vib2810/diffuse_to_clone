FROM osrf/ros:noetic-desktop-full

# add source to bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN apt update && apt install -y git
RUN apt install nano ros-noetic-tf2-tools evince -y
RUN sudo apt install ros-noetic-moveit ros-noetic-franka-ros  -y
RUN sudo apt install python3-pip python3-tk -y
RUN apt install gnome-terminal -y

# install python dependencies to run frankapy within the docker container
RUN pip3 install autolab_core 
RUN pip3 install --force-reinstall pillow==9.0.1 && pip3 install --force-reinstall scipy==1.8
RUN pip3 install numpy-quaternion numba && pip3 install --upgrade google-api-python-client 
RUN pip3 install --force-reinstall numpy==1.23.5

ARG DEBIAN_FRONTEND=noninteractive
# apt install realsense-ros
RUN apt install ros-noetic-realsense2-camera -y
RUN apt update && pip install open3d
RUN apt install ros-noetic-aruco-ros -y
RUN apt-get update && apt-get install -y python3-pip libopenblas-dev libopenmpi-dev
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN apt install ros-noetic-audio-common -y
# Install joy
RUN sudo apt-get install ros-noetic-joy -y

# Install diffusion dependencies
RUN pip3 install torch==1.13.1 torchvision==0.14.1 diffusers==0.18.2 \
        scikit-image==0.19.3 scikit-video==1.1.11 zarr==2.12.0 numcodecs==0.10.2 \
        pygame==2.1.2 pymunk==6.2.1 gym==0.26.2 shapely==1.8.4

# Install pydub and ffmpeg
RUN pip3 install pydub
RUN apt update && apt install ffmpeg -y

# mount src folder from desktop to /home/ros_ws/src
COPY src/il_packages /home/ros_ws/src/il_packages

RUN mkdir -p /home/ros_ws/src/git_packages 
RUN cd /home/ros_ws/src/git_packages && git clone --recursive https://github.com/iamlab-cmu/frankapy.git \
        && git clone https://github.com/ros-planning/panda_moveit_config.git -b noetic-devel \
        && git clone https://github.com/IFL-CAMP/easy_handeye

#rosdep install on src folder
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; cd /home/ros_ws; rosdep install --from-paths src --ignore-src -r -y"

# build workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; cd /home/ros_ws; catkin_make"

RUN echo "source /home/ros_ws/devel/setup.bash" >> ~/.bashrc

# set workdir as home/ros_ws
WORKDIR /home/ros_ws

CMD [ "bash" ]