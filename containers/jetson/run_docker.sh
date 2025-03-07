#!/bin/bash

IMAGE_NAME="dreaming_falcon_jetson"
CONTAINER_NAME="dreaming_falcon_container"
HOST_DIR="/home/eashan/DreamingFalconROS"
CONTAINER_SRC_DIR="/ros2_ws/src/dreaming_falcon"

docker run -it \
    --name "$CONTAINER_NAME" \
    --network host \
    --privileged \
    -v "$HOST_DIR":"$CONTAINER_SRC_DIR" \
    "$IMAGE_NAME" \
    /ros2_ws/src/dreaming_falcon/containers/jetson/build_and_run.sh