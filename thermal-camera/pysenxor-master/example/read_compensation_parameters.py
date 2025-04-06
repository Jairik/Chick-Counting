# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019. All rights reserved.
#
import os, sys
import time
import argparse
import logging
import serial

from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import data_to_frame
from senxor.interfaces import get_serial, USB_Interface

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

# Define an argument parser to convey file with parameters
# Alternatively, define parameters via command line
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--serial_index',
    default=0, type=int,
    help='index of serial port to connect to')
#parser.add_argument('-p', '--parameters',)
args = parser.parse_args()
#print(args)
#sys.exit(0)

# ==============================
# create an USB interface object
# ==============================
try:
    # ser = get_serial()[args.serial_index]
    ser = get_serial()
except IndexError:
    # if on WSL (hack); apply similarly to other cases where
    # device may not be readily found by get_serial
    try:
        ser = serial.Serial('/dev/ttyS4')
    except OSError:
        ser = serial.Serial('/dev/ttyS3')
usb = USB_Interface(ser)
mi48 = MI48([usb, usb])

# print out camera info
camera_info = mi48.get_camera_info()
logger.info('Camera info:')
logger.info(camera_info)

# enable user flash
print('Enabling user flash')
mi48.enable_user_flash()

# read parameters
print('Reading parameters')
rdparams = mi48.get_compensation_params()
print('  '.join([f'{p:.4f}' for p in rdparams]))
sys.exit(0)

# disable user flash to ensure 0x00 reads FF, which is used as a check
# whether we are using a EVK board only or the MI48 board as well.
print('Disabling user flash')
mi48.disable_user_flash()

sys.exit(0)
