#!/bin/bash
#SBATCH --job-name=px4_tuning    # Job name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --mem=8G                # Memory
#SBATCH --time=02:00:00         # Time limit hrs:min:sec
#SBATCH --output=tuning_%j.log  # Standard output and error log

# Check if config index is provided
if [ -z "$1" ]; then
    echo "Usage: sbatch hp_tuning.sh <config_index>"
    exit 1
fi

CONFIG_INDEX=$1

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
    -B ${WORKSPACE_DIR}:/workspace \
    ${CONTAINER} \
    /workspace/container/run_inside_container.sh /workspace/configs/config_${CONFIG_INDEX}.yaml