#!/usr/bin/env bash
set -euxo pipefail

# 1) Debian/RPi packages ───────────────────────────────────
sudo apt update
sudo apt install -y --no-install-recommends \
  build-essential git cmake ninja-build pkg-config \
  python3-dev python3-venv python3-opencv python3-libcamera \
  libcamera-dev libcamera-tools \
  libjpeg-dev libcap-dev libssl-dev \
  i2c-tools \
  python3-pip python3-wheel python3-setuptools \
  python3-numpy python3-smbus python3-spidev  # prebuilt wheels where possible

# 2) create / recreate the venv (includes system packages) ─
cd "$HOME/Chick-Counting"
rm -rf venv
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 3) Python packages ────────────────────────────────────────
pip install --upgrade \
  cmapy==0.6.6 \
  crcmod==1.7 \
  gpiozero==2.0.1 \
  matplotlib==3.10.1 \
  pyserial==3.5 \
  sphinxcontrib-programoutput==0.18

# 4) Rebuild simplejpeg cleanly ─────────────────────────────
pip install --force-reinstall simplejpeg

# 5) Install Picamera2 properly ─────────────────────────────
git clone --depth 1 https://github.com/raspberrypi/picamera2.git tmp_picamera2
pip install ./tmp_picamera2
rm -rf tmp_picamera2

# 6) Done ─ print confirmation ─────────────────────────────
python - <<'PY'
import numpy, simplejpeg, picamera2
print("NumPy      :", numpy.__version__)
print("simplejpeg :", simplejpeg.__file__)
print("Picamera2  :", picamera2.__version__)
PY

echo -e "\n✅  All dependencies built and installed."
echo "Activate the environment any time with:"
echo "  source ~/Chick-Counting/venv/bin/activate"
echo "Then run:"
echo "  python DUAL_VIDEO_CAPTURE.py"

