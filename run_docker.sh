xhost +local:root 
docker container prune -f 
docker run --privileged --rm -it \
    --name="diffusion" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTH:$XAUTH" \
    --network host \
    -v "$(pwd)/src/il_packages:/home/ros_ws/src/il_packages" \
    -v "$(pwd)/networks:/home/ros_ws/networks" \
    -v "$(pwd)/guide_mode.py:/home/ros_ws/guide_mode.py" \
    -v "$(pwd)/bags:/home/ros_ws/bags" \
    -v "$(pwd)/dataset:/home/ros_ws/dataset" \
    -v "/etc/timezone:/etc/timezone:ro" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    --gpus all \
    diffusion bash