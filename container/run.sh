#!/bin/bash

# Mount DreamingFalconROS next to PX4-Autopilot in workspace
apptainer run \
    -B ${PWD}:/workspace/DreamingFalconROS \
    dreamingfalcon.sif "$@"