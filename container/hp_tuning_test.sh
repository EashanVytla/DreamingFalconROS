#!/bin/bash

# Check if config index is provided
if [ -z "$1" ]; then
    echo "Usage: sbatch hp_tuning.sh <config_index>"
    exit 1
fi

CONFIG_INDEX=$1

# Directory setup
WORKSPACE_DIR="${HOME_DIR}/workspace"
CONFIG_DIR="${WORKSPACE_DIR}/DreamingFalconROS/configs"
CONTAINER="${WORKSPACE_DIR}/DreamingFalconROS/container/dreamingfalcon.sif"
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_INDEX}.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist"
    exit 1
fi

# Make the script executable
chmod +x ${WORKSPACE_DIR}/DreamingFalconROS/container/run_inside_container.sh

# Run the container with the script and pass the config file
apptainer exec \
    --fakeroot \
    --nv \
    -B ${WORKSPACE_DIR}:/workspace \
    ${CONTAINER} \
    cd /workspace/DreamingFalconROS \
    /workspace/container/run_inside_container.sh /workspace/configs/config_${CONFIG_INDEX}.yaml