#!/bin/bash

source /opt/ros/humble/setup.bash
if [ -f "/ros2_ws/DreamingFalconROS/install/setup.bash" ]; then
  source /ros2_ws/DreamingFalconROS/install/setup.bash
fi

exec "$@"