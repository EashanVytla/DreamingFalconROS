#!/bin/bash

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
PORT="/dev/ttyUSB0"

# Create logs directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# LOG_DIR="/ros2_ws/logs/run_${TIMESTAMP}"
LOG_DIR="/ros2_ws/logs"
mkdir -p "${LOG_DIR}"

# MicroXRCE Agent
echo "Starting MicroXRCEAgent"
MicroXRCEAgent serial --dev $PORT -b 921600 > "${LOG_DIR}/microxrce.log" 2>&1 &

# Navigate to workspace
cd /ros2_ws/DreamingFalconROS

# Start TensorBoard
echo "Starting TensorBoard"
tensorboard --logdir=runs --bind_all > "${LOG_DIR}/tensorboard.log" 2>&1 &

# Build packages
echo -e "${YELLOW}Building ROS packages...${NC}"
colcon build

# Source workspace
source /ros2_ws/DreamingFalconROS/install/local_setup.bash

# Launch application
echo -e "${GREEN}Build complete. Starting ROS application...${NC}"
pwd
ls /ros2_ws/DreamingFalconROS/install/px4_ros_com/lib/px4_ros_com
ros2 launch px4_ros_com chirp.launch.py 2>&1 | tee "${LOG_DIR}/ros.log"
