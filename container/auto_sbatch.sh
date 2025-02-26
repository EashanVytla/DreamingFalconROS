#!/bin/bash
#SBATCH --account=pas2152
#SBATCH --job-name=auto_sbatch    # Job name
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=8:30:00         # Time limit hrs:min:sec
#SBATCH --output=auto_sbatch.log  # Standard output and error log
#SBATCH --mail-type=BEGIN,FAIL

# Directory setup
WORKSPACE_DIR=$HOME/workspace
CONFIG_DIR="${WORKSPACE_DIR}/DreamingFalconROS/configs"
COMPLETED_DIR="${WORKSPACE_DIR}/DreamingFalconROS/configs/completed"

# Create completed directory if it doesn't exist
mkdir -p "${COMPLETED_DIR}"

# Make scripts executable
chmod +x ${WORKSPACE_DIR}/DreamingFalconROS/container/hp_tuning.sh
chmod +x ${WORKSPACE_DIR}/DreamingFalconROS/container/run_inside_container.sh

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
        # Skip if already in completed directory
        filename=$(basename "$config_file")
        if [ -f "${COMPLETED_DIR}/${filename}" ]; then
            echo "Skipping ${filename} - already completed"
            continue
        fi  # Fixed syntax: removed }

        # Extract index from filename
        config_index=$(get_config_index "$config_file")
        
        echo "Submitting job for config_${config_index}.yaml"
        
        # Submit the job and capture the job ID
        job_id=$(sbatch --parsable ${WORKSPACE_DIR}/DreamingFalconROS/container/hp_tuning.sh "$config_index")
        
        echo "Submitted job ${job_id} for config_${config_index}.yaml"
        
        # Optional: add a delay between submissions (22 minutes)
        sleep 1320
    fi
done

echo "All jobs submitted. Monitor with 'squeue -u $USER'"