#!/bin/bash
# Will record until script is exited with exit signal

FILENAME="video_$(date +%Y-%m-%d_%H-%M-%S).h264"

echo "Recording... Press Ctrl+C to stop."

# Start recording indefinitely
libcamera-vid -o "$FILENAME" --width 1920 --height 1080 --framerate 30 -t 0

echo "Video saved as $FILENAME"

