#!/bin/bash
# Test program to capture an image indefinitely every 5 seconds. Adding date time stamp

while true; do
    FILENAME="capture_$(date +%Y-%m-%d_%H-%M-%S).jpg"
    # MAIN MOVER: This function captures the image, under the specified filename, width, and height
    libcamera-jpeg -o "$FILENAME" --width 1920 --height 1080
    echo "Picture saved as $FILENAME"
    sleep 5  # Wait 5 seconds before taking another picture
done

