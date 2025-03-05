#!/bin/bash
# Test program that simply takes a picture - used for demo

FILENAME="capture_$(date +%Y-%m-%d_%H-%M-%S).jpg"
libcamera-jpeg -o "$FILENAME" --width 1920 --height 1080
echo "Picture saved as $FILENAME"

# TO ACCESS - feh <picture name>
