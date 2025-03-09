#!/bin/bash
# filepath: /home/eashan/DreamingFalconROS/containers/jetson/build_and_run.sh

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to workspace
cd /ros2_ws/DreamingFalconROS

# Build packages
echo -e "${YELLOW}Building ROS packages...${NC}"
colcon build --symlink-install

# Source workspace
source /ros2_ws/DreamingFalconROS/install/setup.bash

# Launch application
echo -e "${GREEN}Build complete. Starting ROS application...${NC}"
ros2 launch dreaming_falcon px4_ros_com_launch.py
