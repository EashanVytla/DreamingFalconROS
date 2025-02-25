#!/bin/bash

CONFIG_FILE=$1
CONFIG_INDEX=$(basename "$CONFIG_FILE" | sed 's/config_\(.*\)\.yaml/\1/')
# WORKSPACE_DIR=${HOME}
WORKSPACE_DIR="/workspace"
LOG_DIR="${WORKSPACE_DIR}/logs/run_${CONFIG_INDEX}"
COMPLETED_DIR="${WORKSPACE_DIR}/DreamingFalconROS/configs/completed"
TIMEOUT=1260
BUILD_PATH="$WORKSPACE_DIR/PX4-Autopilot/build/px4_sitl_default"

export HEADLESS=1

# Create log directories
mkdir -p "${LOG_DIR}"/{px4,agent,chirp}

echo "Starting PX4 Autopilot..."
cd ${WORKSPACE_DIR}/PX4-Autopilot
if ! $BUILD_PATH/bin/px4 -d > "$LOG_DIR/px4/px4_sitl.log" 2> "$LOG_DIR/px4/px4_error.log" & then
    PX4_PID=$!
    echo "Started PX4 Autopilot with PID: $PX4_PID"
else
    echo "Failed to start PX4 Autopilot"
    exit 1
fi

timeout=30
elapsed=0
ready=false
while [ $elapsed -lt $timeout ]; do
    if grep -q "Ready for takeoff!" "$LOG_DIR/px4/px4_sitl.log"; then
        ready=true
        echo "PX4 is ready for takeoff!"
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    echo "Waiting for PX4 to be ready... ${elapsed}/${timeout} seconds"
done

if [ "$ready" = false ]; then
    echo "ERROR: PX4 failed to become ready within ${timeout} seconds"
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
source ${WORKSPACE_DIR}/DreamingFalconROS/install/local_setup.bash

cd ${WORKSPACE_DIR}/DreamingFalconROS
source .venv/bin/activate
# Run chirp launch file with specified config and logging
echo "Running with configuration: ${CONFIG_FILE}"
ros2 launch px4_ros_com chirp.launch.py config_file:=${WORKSPACE_DIR}/DreamingFalconROS/${CONFIG_FILE} \
    > "${LOG_DIR}/chirp/chirp.log" 2>&1

# Wait for any remaining background processes
wait

echo "Tuning run completed. Logs available in ${LOG_DIR}"

# Move config file to completed directory
mv "${WORKSPACE_DIR}/DreamingFalconROS/config_${CONFIG_FILE}.yaml" "${COMPLETED_DIR}/"
echo "Moved config_${CONFIG_FILE}.yaml to completed directory"

# Let the cleanup() function handle process termination via the EXIT trap
exit 0