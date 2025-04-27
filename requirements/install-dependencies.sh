#!/bin/bash

# Update APT repositories
echo "Updating package lists..."
sudo apt update

# Install APT dependencies
echo "Installing required APT packages..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libcap-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk* \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libblas-dev \
    libqtgui4 \
    libqt4-test \
    libilmbase-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgdk-pixbuf2.0-dev \
    libtbb2 \
    libtbb-dev \
    zlib1g-dev

# Raspberry Pi-specific libraries if on a Pi
sudo apt install -y libcamera-dev libcamera-apps

# Now install Python packages from requirements.txt
echo "Installing required Python packages via pip..."
pip install --upgrade pip
pip install -r requirements.txt

# Potentially redundant, but ensures all dependencies are successfully installed
pip install numpy simplejpeg opencv-python

# Install Picamera2 manually
git clone https://github.com/raspberrypi/picamera2.git
cd picamera2
pip install .

echo "All installations complete!"

