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
parser.add_argument('-f', '--param_file',
    default=os.path.expanduser('~/Meridian/Production/'+\
    'xpro-unit-calib-params.csv'), type=str,
    help='file with compensation parameters per module SN')

parser.add_argument('-rst', '--reset_parameters',
                    default=False, action='store_true',
                    help='Clear all compensation parameters to 0')

#parser.add_argument('-p', '--parameters', )

args = parser.parse_args()
#print(args)
#sys.exit(0)

# ==============================
# create an USB interface object
# ==============================
try:
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

# initialize parameters
# these must always be 4 and different than <0xFFFFFFFF
params = [0, 0, 0, 0]

# fetch parameters from file
if not args.reset_parameters:
    with open(args.param_file, 'r') as f:
#        header_line = f.readline()
        for line in f.readlines():
            words = line.split(',')
            sn = words[0]
            if mi48.sn.upper() == sn.upper():
                params = [float(w) for w in words[1:]]
                # ensure that we're updating all four parameters
                # if file provides less than 4 params, we must still clear
                # the FFs from the Flash addresses that store all 4 params.
                for i in range(4 - len(params)):
                    params.append(0)
                break

    if not params:
        print(f'No parameters found for {mi48.sn} in {args.param_file}')
        sys.exit(1)

    print(f'Found compensation parameters for {mi48.sn}:')
    print('  '.join([f'{p:.4f}' for p in params]))

else:
    print(f'Resetting compensation parameters for {mi48.sn}: to ', params)

# sys.exit(0)


# enable user flash
print('Writing parameters to the MI48 user flash')
mi48.enable_user_flash()
mi48.store_compensation_params(params)

# read them back
print('Verifying parameters')
rdparams = mi48.get_compensation_params()
print('  '.join([f'{p:.4f}' for p in rdparams]))

# disable user flash to ensure 0x00 reads FF, which is used as a check
# whether we are using a EVK board only or the MI48 board as well.
print('Disabling user flash')
mi48.disable_user_flash()

sys.exit(0)
