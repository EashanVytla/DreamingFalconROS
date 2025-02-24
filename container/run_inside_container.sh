#!/bin/bash

CONFIG_FILE=$1
CONFIG_INDEX=$(basename "$CONFIG_FILE" | sed 's/config_\(.*\)\.yaml/\1/')
# WORKSPACE_DIR=".."
WORKSPACE_DIR="/workspace"
LOG_DIR="${WORKSPACE_DIR}/logs/run_${CONFIG_INDEX}"
TIMEOUT=180

# Create log directories
mkdir -p "${LOG_DIR}"/{px4,agent,chirp}

# Function to start PX4 SITL
start_px4_sitl() {
    cd ${WORKSPACE_DIR}/PX4-Autopilot
    HEADLESS=1 make px4_sitl jmavsim > "${LOG_DIR}/px4/px4_sitl.log" 2>&1 &
    PX4_PID=$!
    echo "Started PX4 SITL with PID: $PX4_PID"
    
    # Simple 20 second wait
    for i in {1..20}; do
        if ! kill -0 $PX4_PID 2>/dev/null; then
            echo "PX4 SITL process died unexpectedly"
            return 1
        fi
        echo "Waiting for PX4 SITL... ($i/20s)"
        sleep 1
    done

    # Check if process is still running after wait
    if kill -0 $PX4_PID 2>/dev/null; then
        echo "PX4 SITL startup complete"
        return 0
    else
        echo "PX4 SITL process died"
        return 1
    fi
}

# Replace existing PX4 SITL start section with:
max_retries=3
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    start_px4_sitl
    if [ $? -eq 0 ]; then
        echo "PX4 SITL started successfully"
        break
    fi
    retry_count=$((retry_count + 1))
    echo "Retry $retry_count of $max_retries"
done

if [ $retry_count -eq $max_retries ]; then
    echo "Failed to start PX4 SITL after $max_retries attempts"
    cleanup
    exit 1
fi

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

# Run chirp launch file with specified config and logging
echo "Running with configuration: ${CONFIG_FILE}"
ros2 launch px4_ros_com chirp.launch.py config_file:=${WORKSPACE_DIR}/DreamingFalconROS/${CONFIG_FILE} \
    > "${LOG_DIR}/chirp/chirp.log" 2>&1

# Wait for any remaining background processes
wait

echo "Tuning run completed. Logs available in ${LOG_DIR}"