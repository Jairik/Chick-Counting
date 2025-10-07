#!/bin/bash

# Set filename with timestamp
FILENAME="video_$(date +%Y-%m-%d_%H-%M-%S).h264"

# Specify video duration here, will take in as an argument
DURATION=5

# MAIN MOVER: Main command that specifies filename, width, height, framerate, and duration
libcamera-vid -o "$FILENAME" --width 1920 --height 1080 --framerate 30 -t "$DURATION"

# Print success message to confirm that video was taken
echo "Video saved as $FILENAME"

