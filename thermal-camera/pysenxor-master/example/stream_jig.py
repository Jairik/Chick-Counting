# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2020. All rights reserved.
#
import sys
import os
import signal
import time
import logging
import serial
import cv2 as cv

from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import data_to_frame, remap, cv_filter, cv_render
from senxor.interfaces import MI_VID, MI_PIDs, USB_Interface
import serial
from serial.tools import list_ports

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))


# Make the a global variable and use it as an instance of the mi48.
# This allows it to be used directly in a signal_handler.
global mi48

# define a signal handler to ensure clean closure upon CTRL+C
# or kill from terminal
def signal_handler(sig, frame):
    """Ensure clean exit in case of SIGINT or SIGTERM"""
    logger.info("Exiting due to SIGINT or SIGTERM")
    mi48.stop()
    cv.destroyAllWindows()
    logger.info("Done.")
    sys.exit(0)

# Define the signals that should be handled to ensure clean exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==============================
# create an USB interface object
# ==============================
def connect_senxor(socket_id=13, raw=False):
    """
    Attempt to connect to senxor, return an instance of the MI48.

    `socket_id` is the jig socket, enumerated from 1 - top left, to 25 - bottom right.

    If `raw`, then the stream will contain uint16 deci-Kelvin,
    else the stream will be float temperature in Celsius.
    """
    dev_list = []
    vcom_list = []
    for p in list_ports.comports():
        if p.vid == MI_VID and p.pid in MI_PIDs:
            dev_list.append(p.device)
            vcom_list.append(p.description.split()[-1][1:-1])

    if not dev_list:
        raise RuntimeError('No SenXor connected')

    for dev, com in zip(dev_list, vcom_list):
        try:
            ser = serial.Serial(dev)
            ser.baudrate = 115200
            ser.rtscts = True
            ser.dsrdtr = True
            ser.timeout = 0.5
            ser.write_timeout = 0.5
            usb = USB_Interface(ser)
            mi48 = MI48([usb,usb], name=com, read_raw=raw)
            mi48.powerup()
            time.sleep(0.5)
            sid = mi48.get_evk_socket_id()
            logger.info(f'socket_id {sid}, port {com}')
            if sid  == socket_id: break
            else: continue
        except serial.SerialException:
            logger.critical(f'Failed opening {dev}')
            raise
    # here we already have gotten a serial device; set it up
    logging.debug(f'{mi48.sn} in socket {socket_id} connected to {com}')
    mi48.get_camera_info()
    return mi48


mi48 = connect_senxor(socket_id=18)

#sys.exit(0)


# set desired FPS
if len(sys.argv) == 2:
    STREAM_FPS = int(sys.argv[1])
else:
    STREAM_FPS = 9
mi48.set_fps(STREAM_FPS)

# see if filtering is available in MI48 and set it up
mi48.disable_filter(f1=True, f2=True, f3=True)
mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
mi48.set_offset_corr(0.0)

# initiate continuous frame acquisition
with_header = True
mi48.start(stream=True, with_header=with_header)

# change this to false if not interested in the image
GUI = True

# set cv_filter parameters
par = {'blur_ks':5, 'd':5, 'sigmaColor': 27, 'sigmaSpace': 27}

while True:
    data, header = mi48.read()
    if data is None:
        logger.critical('NONE data received instead of GFRA')
        mi48.stop()
        sys.exit(1)

    frame = data_to_frame(data, (80,62), hflip=True);
    filt_uint8 = cv_filter(remap(frame), par, use_median=True,
                           use_bilat=True, use_nlm=False)
    #
    if header is not None:
        logger.debug('  '.join([format_header(header),
                                format_framestats(data)]))
    else:
        logger.debug(format_framestats(data))

    if GUI:
        cv_render(filt_uint8, resize=(400,310))
        key = cv.waitKey(1)  # & 0xFF
        if key == ord("q"):
            break
#    time.sleep(1)

# stop capture and quit
mi48.stop()
cv.destroyAllWindows()
