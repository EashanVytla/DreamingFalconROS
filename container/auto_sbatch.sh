#!/bin/bash

# Directory setup
WORKSPACE_DIR="/home/eashan/DreamingFalconROS"
CONFIG_DIR="${WORKSPACE_DIR}/configs"

# Check if configs directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Configs directory does not exist at $CONFIG_DIR"
    exit 1
fi

# Function to extract index from config filename
get_config_index() {
    local filename=$(basename "$1")
    echo "$filename" | sed 's/config_\(.*\)\.yaml/\1/'
}

# Loop through all config files
for config_file in ${CONFIG_DIR}/config_*.yaml; do
    if [ -f "$config_file" ]; then
        # Extract index from filename
        config_index=$(get_config_index "$config_file")
        
        echo "Submitting job for config_${config_index}.yaml"
        
        # Submit the job and capture the job ID
        job_id=$(sbatch --parsable ${WORKSPACE_DIR}/container/hp_tuning.sh "$config_index")
        
        echo "Submitted job ${job_id} for config_${config_index}.yaml"
        
        # Optional: add a delay between submissions
        sleep 2
    fi
done

echo "All jobs submitted. Monitor with 'squeue -u $USER'"