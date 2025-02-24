#!/bin/bash

CONFIG_FILE=$1
CONFIG_INDEX=$(basename "$CONFIG_FILE" | sed 's/config_\(.*\)\.yaml/\1/')
# WORKSPACE_DIR=${HOME}
WORKSPACE_DIR="/workspace"
LOG_DIR="${WORKSPACE_DIR}/logs/run_${CONFIG_INDEX}"
TIMEOUT=1260
WAIT_FOR_PX4=30

# Create log directories
mkdir -p "${LOG_DIR}"/{px4,agent,chirp}

echo "Starting PX4 Autopilot..."
cd ${WORKSPACE_DIR}/PX4-Autopilot
HEADLESS=1 make px4_sitl jmavsim > "${LOG_DIR}/px4/px4_sitl.log" 2>&1 &
elapsed=0
while [ $elapsed -lt $WAIT_FOR_PX4 ]; do
    echo "Time elapsed: ${elapsed}/${WAIT_FOR_PX4} seconds"
    sleep 1
    elapsed=$((elapsed + 1))
done

# Function to kill background processes on script exit
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "px4_sitl"
    pkill -f "MicroXRCEAgent"
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup EXIT

# Start timeout monitor in background
(
    elapsed=0
    while [ $elapsed -lt $TIMEOUT ]; do
        echo "Time elapsed: ${elapsed}/${TIMEOUT} seconds"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "Timeout reached ($TIMEOUT seconds). Initiating shutdown..."
    cleanup
) &
TIMEOUT_PID=$!

# Start MicroXRCE Agent with logging
MicroXRCEAgent udp4 -p 8888 > "${LOG_DIR}/agent/microros_agent.log" 2>&1 &
AGENT_PID=$!
sleep 5

# Source ROS environment
source /opt/ros/humble/setup.bash
source ${WORKSPACE_DIR}/DreamingFalconROS/install/setup.bash

cd ${WORKSPACE_DIR}/DreamingFalconROS
# Run chirp launch file with specified config and logging
echo "Running with configuration: ${CONFIG_FILE}"
ros2 launch px4_ros_com chirp.launch.py config_file:=${WORKSPACE_DIR}/DreamingFalconROS/${CONFIG_FILE} \
    > "${LOG_DIR}/chirp/chirp.log" 2>&1

# Wait for any remaining background processes
wait

echo "Tuning run completed. Logs available in ${LOG_DIR}"