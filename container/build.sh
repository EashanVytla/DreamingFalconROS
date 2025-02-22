#!/bin/bash

# Set error handling
set -e

# Define variables
CONTAINER_DEF="container.def"
OUTPUT_NAME="dreamingfalcon.sif"
BUILD_LOG="build_log.txt"

# Check if definition file exists
if [ ! -f "$CONTAINER_DEF" ]; then
    echo "Error: $CONTAINER_DEF not found!"
    exit 1
fi

echo "Starting Apptainer build process..."

# Build the container
sudo apptainer build "$OUTPUT_NAME" "$CONTAINER_DEF" 2>&1 | tee "$BUILD_LOG"

if [ $? -eq 0 ]; then
    echo "Container built successfully!"
    echo "Output container: $OUTPUT_NAME"
    echo "Build log: $BUILD_LOG"
else
    echo "Error: Container build failed. Check $BUILD_LOG for details."
    exit 1
fi