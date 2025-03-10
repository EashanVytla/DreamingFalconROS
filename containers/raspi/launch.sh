#!/bin/bash

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to workspace
cd /ros2_ws/DreamingFalconROS

# Build packages
echo -e "${YELLOW}Building ROS packages...${NC}"
colcon build

# Source workspace
source /ros2_ws/DreamingFalconROS/install/local_setup.bash

# Launch application
echo -e "${GREEN}Build complete. Starting ROS application...${NC}"
pwd
ls /ros2_ws/DreamingFalconROS/install/px4_ros_com/lib/px4_ros_com
ros2 launch px4_ros_com chirp.launch.py
