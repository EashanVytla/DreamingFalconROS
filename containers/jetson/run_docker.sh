#!/bin/bash

IMAGE_NAME="dreamer_docker"
CONTAINER_NAME="dreamer_container"
HOST_DIR="/home/eashan/ros2_ws"
CONTAINER_SRC_DIR="/ros2_ws"

docker run -it \
    --name "$CONTAINER_NAME" \
    --network host \
    --privileged \
    -v "$HOST_DIR":"$CONTAINER_SRC_DIR" \
    "$IMAGE_NAME" \
    /ros2_ws/DreamingFalconROS/containers/jetson/launch.sh
