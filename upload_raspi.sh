#!/bin/bash
# filepath: /home/eashan/DreamingFalconROS/upload_raspi.sh

# Define source and destination
SOURCE_DIR="$(pwd)"
REMOTE_HOST="eashan@192.168.1.1"
REMOTE_DIR="~/ros2_ws/DreamingFalconROS"  # Without the ~ since SSH will expand it

# First ensure the target directory exists
echo "Ensuring target directory exists..."
ssh ${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}"

# Use rsync instead of scp for better control
echo "Copying files to ${REMOTE_HOST}:${REMOTE_DIR}..."
rsync -avz --progress \
    --exclude="build/" \
    --exclude="install/" \
    --exclude="log/" \
    --exclude=".git/" \
    --exclude="venv/" \
    --exclude="containers/" \
    "${SOURCE_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Files copied successfully!"
else
    echo "Error: Failed to copy files"
    exit 1
fi