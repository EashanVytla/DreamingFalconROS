#!/bin/bash

# Check if config index is provided
if [ -z "$1" ]; then
    echo "Usage: sbatch hp_tuning.sh <config_index>"
    exit 1
fi

CONFIG_INDEX=$1

#SBATCH --account=pas2152
#SBATCH --job-name=wm_training_exp_${CONFIG_INDEX}    # Job name
#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --time=00:30:00         # Time limit hrs:min:sec
#SBATCH --output=tuning_${CONFIG_INDEX}_%j.log  # Standard output and error log
#SBATCH --mail-type=ALL

# Directory setup
WORKSPACE_DIR="/home/eashan/DreamingFalconROS"
CONFIG_DIR="${WORKSPACE_DIR}/configs"
CONTAINER="${WORKSPACE_DIR}/container/dreamingfalcon2.sif"
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_INDEX}.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist"
    exit 1
fi

# Make the script executable
chmod +x ${WORKSPACE_DIR}/container/run_inside_container.sh

# Run the container with the script and pass the config file
apptainer exec \
    --fakeroot \
    --nv \
    -B ${WORKSPACE_DIR}:/workspace \
    ${CONTAINER} \
    cd /workspace/DreamingFalconROS \
    /workspace/container/run_inside_container.sh /workspace/configs/config_${CONFIG_INDEX}.yaml