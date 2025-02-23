#!/bin/bash

CONFIG_FILE=$1
CONFIG_INDEX=$(basename "$CONFIG_FILE" | sed 's/config_\(.*\)\.yaml/\1/')
LOG_DIR="/workspace/logs/run_${CONFIG_INDEX}"
TIMEOUT=1260  # 21 minutes in seconds

# Create log directories
mkdir -p "${LOG_DIR}"/{px4,agent,chirp}

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
    sleep $TIMEOUT
    echo "Timeout reached (21 minutes). Initiating shutdown..."
    cleanup
) &
TIMEOUT_PID=$!

# Start PX4 SITL simulation with logging
cd /workspace/PX4-Autopilot
HEADLESS=1 make px4_sitl jmavsim > "${LOG_DIR}/px4/px4_sitl.log" 2>&1 &
PX4_PID=$!
sleep 20

# Start MicroXRCE Agent with logging
MicroXRCEAgent udp4 -p 8888 > "${LOG_DIR}/agent/microros_agent.log" 2>&1 &
AGENT_PID=$!
sleep 5

# Source ROS environment
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Run chirp launch file with specified config and logging
echo "Running with configuration: ${CONFIG_FILE}"
ros2 launch /workspace/launch/chirp.launch.py config_file:=${CONFIG_FILE} \
    > "${LOG_DIR}/chirp/chirp.log" 2>&1

# Kill the timeout process if we finished before it
kill $TIMEOUT_PID 2>/dev/null

# Wait for any remaining background processes
wait

echo "Tuning run completed. Logs available in ${LOG_DIR}"