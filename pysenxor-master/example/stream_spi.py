# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019. All rights reserved.
#
import sys
import os
import signal
from smbus import SMBus
from spidev import SpiDev
import argparse

try:
    from gpiozero import Pin, DigitalInputDevice, DigitalOutputDevice
except:
    import sys
    print("Please install the 'gpiozero' library to monitor "
          "the MI48 DATA_READY pin. For example, by:")
    print("pip3 install gpiozero")
    sys.exit()

import time
import logging
import numpy as np

import cv2 as cv

from senxor.mi48 import MI48, DATA_READY, format_header, format_framestats
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record', default=False, dest='record',
                        action='store_true', help='Record data'),
    parser.add_argument('-fps', '--framerate', default=7,
                        type=float, help='Bobcat framerate', dest='fps')
    parser.add_argument('-c', '--colormap', default='rainbow2', type=str,
                        help='Colormap')
    args = parser.parse_args()
    return args


def get_filename(tag, ext=None):
    """Yield a timestamped filename with specified tag."""
    ts = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    filename = "{}--{}".format(tag, ts)
    if ext is not None:
        filename += '.{}'.format(ext)
    return filename

def write_frame(outfile, arr):
    """Write a numpy array as a row in a file, using C ordering.

    Assume that filename is already created and available.
    """
    #
    if arr.dtype == np.uint16:
        outstr = ('{:n} '*arr.size).format(*arr.ravel(order='C')) + '\n'
    else:
        outstr = ('{:.2f} '*arr.size).format(*arr.ravel(order='C')) + '\n'
    try:
        # assume outfile is a handle to a file open for write
        outfile.write(outstr)
        # we have a relatively large outstr (~5K * 7ASCII chars per pixel)
        # since the OS has by default ~8KB buffer, it will be good to
        # flush so as to not output incomplete frame to the file
        # (which may happen if we early terminate for some reason)
        # or else it may lead to partially output frame and problems
        # upon read.
        outfile.flush()
        return None
    except AttributeError:
        # assume outfile is a name to a file
        # this should automatically flush the buffer
        with open(outfile, 'a') as fh:
            fh.write(outstr)
        return None
    except IOError:
        logger.critical('Cannot write to {} (IOError)'.format(outfile))
        sys.exit(106)

def cv_display(img, title='', resize=(320, 248),
               colormap=cv.COLORMAP_JET, interpolation=cv.INTER_CUBIC):
#               colormap=cv.COLORMAP_JET, interpolation=cv.INTER_LINEAR):
    """
    Display image using OpenCV-controled window.

    Data is a 2D numpy array of type uint8,

    Image is coloured and resized
    """
    cvcol = cv.applyColorMap(img, colormap)
    cvresize =  cv.resize(cvcol, resize, interpolation=interpolation)
    cv.imshow(title, cvresize)


# Main starts here; ideally we shall isolate that in a function
# -------------------------------------------------------------
#
# Parse command line arguments
args = parse_args()

# ls /dev/*i2c* should indicate which i2c channels are available to user space
RPI_GPIO_I2C_CHANNEL = 1

# ls /dev/*spi*
# should indicate which spi bus and what devices are available:
# e.g. /dev/spidev<bus>.<device>
# /dev/spidev0.0  /dev/spidev0.1
RPI_GPIO_SPI_BUS = 0

# MI48A CS is routed to CE1 of the RPI on the uHAT development board
# adapt that value according to your setup
RPI_GPIO_SPI_CE_MI48 = 1

# =======================
# MI48 I2C Address:
# =======================
# could be 0x40 or 0x41, depending on how ADDR pin of the chip is tied.
# use
# $i2cdetect -y 1
# on the command prompt to confirm which address the device responds to
MI48_I2C_ADDRESS = 0x40

# =======================
# MI48 SPI Stuff:
# =======================
MI48_SPI_MODE = 0b00
MI48_SPI_BITS_PER_WORD = 8   # cannot be handled with /dev/spidev-x.y and python on RPi 3B+; must work with default 8
MI48_SPI_LSBFIRST = False    # this appears to be a read-only on RPi
MI48_SPI_CSHIGH = True
# MI48_SPI_MAX_SPEED_HZ = 7800000
# MI48_SPI_MAX_SPEED_HZ = 15600000
MI48_SPI_MAX_SPEED_HZ = 31200000
MI48_SPI_CS_DELAY = 0.0001   # delay between asserting/deasserting CS_N and initiating/stopping clock/data


def close_all_interfaces():
    try:
        spi.close()
    except NameError:
        pass
    try:
        i2c.close()
    except NameError:
        pass

# define a signal handler to ensure clean closure upon CTRL+C
# or kill from terminal
def signal_handler(sig, frame):
    """Ensure clean exit in case of SIGINT or SIGTERM"""
    logger.info("Exiting due to SIGINT or SIGTERM")
    mi48.stop(poll_timeout=0.25, stop_timeout=1.2)
    time.sleep(0.5)
    cv.destroyAllWindows()
    logger.info("Done.")
    sys.exit(0)

# Define the signals that should be handled to ensure clean exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# create an I2C interface object
i2c = I2C_Interface(SMBus(RPI_GPIO_I2C_CHANNEL), MI48_I2C_ADDRESS)

# ==============================
# Create an SPI interface object
# ==============================
#
# One needs to chose a buffer size for transfer; Optimal size may be
# different depending on target FPS and host's resources
#
# cat /sys/module/spidev/parameters/bufsiz to check default size
# Default size can potentially be changed via /boot/cmdline.txt on RPI
# spidev.bufsize=<NEEDED BUFFER SIZE>
# Preferred way may be with the initialisation of the spi object.
# We chose 160 bytes which corresponds to 1 row on MI08xx
SPI_XFER_SIZE_BYTES = 160  # bytes
spi = SPI_Interface(SpiDev(RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48),
                    xfer_size=SPI_XFER_SIZE_BYTES)

spi.device.mode = MI48_SPI_MODE
spi.device.max_speed_hz = MI48_SPI_MAX_SPEED_HZ
spi.device.bits_per_word = 8
spi.device.lsbfirst = False   # seems to be a read-only value;
                              # likely reflecting cpu endianness
#spi.device.cshigh = MI48_SPI_CSHIGH
# in linux kernel 5.x.x+ ioctl module does not handle the SPI CS
# any more, since it is thought that it is a devcie property,
# not a bus property. We therefore leave the CS to a GPIO handling.
# Note that on the uHat board that we have with MI48 and Bobcat,
# the CS is on GPIO-7 (J8 connector Pin 26).
spi.device.cshigh = True
spi.device.no_cs = True
mi48_spi_cs_n = DigitalOutputDevice("BCM7", active_high=False,
                                    initial_value=False)


# ===============================
# Configure DATA_READY and RESET
# ===============================

# Assuming that DATA_READY is connected
# NOTABENE:
# The MI48.DATA_READY pin is routed to BROADCOM.GPIO.24,
# which is pin 18 on the 40-pin header.
# The gpiozero library uses the BROADCOM convention, hence we have
# "BCM24" below, or just 24.
#
# Change this to False to test DATA_READY flag, instead of pin
use_data_ready_pin = True
if use_data_ready_pin:
    mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False)

# connect the reset line to allow to drive it by SW (GPIO23, J8:16)
mi48_reset_n = DigitalOutputDevice("BCM23", active_high=False,
                                   initial_value=True)

class MI48_reset:
    def __init__(self, pin,
                 assert_seconds=0.000035,
                 deassert_seconds=0.050):
        self.pin = pin
        self.assert_time = assert_seconds
        self.deassert_time = deassert_seconds

    def __call__(self):
        print('Resetting the MI48...')
        self.pin.on()
        time.sleep(self.assert_time)
        self.pin.off()
        time.sleep(self.deassert_time)
        print('Done.')

# ==============================
# Create an MI48 interface object
# ==============================
mi48 = MI48([i2c, spi], data_ready=mi48_data_ready,
            reset_handler=MI48_reset(pin=mi48_reset_n))

# print out camera info
camera_info = mi48.get_camera_info()
logger.info('Camera info:')
logger.info(camera_info)

# set desired FPS
# TODO: investigate issue at > 9 FPS on R-Pi 3B+
mi48.set_fps(args.fps)

# see if filtering is available in MI48 and set it up
if int(mi48.fw_version[0]) >= 2:
    # Enable filtering with default strengths
    mi48.enable_filter(f1=True, f2=True, f3=False)

    # If needed, set a temperature offset across entire frame
    # e.g. if overall accuracy (at product level) seems to be 
    # 0.7 above the blackbody, then we need to subtract 0.7 
    # from the readout of the MI48:
    # mi48.set_offset_corr(-5.55)
    #
    # However, for most applications the factory level, per pixel
    # calibration is sufficient, so keep offset 0
    mi48.set_offset_corr(0.0)

# initiate continuous frame acquisition
with_header = True

# enable saving to a file
if args.record:
    filename = get_filename(mi48.camera_id_hex)
    fd_data = open(os.path.join('.', filename+'.dat'), 'w')

mi48.start(stream=True, with_header=with_header)

# change this to false if not interested in the image
GUI = True

while True:
    # wait for data_ready pin (or poll for STATUS.DATA_READY /fw 2.1.X+)
    if hasattr(mi48, 'data_ready'):
        mi48.data_ready.wait_for_active()
    else:
        data_ready = False
        while not data_ready:
            time.sleep(0.01)
            data_ready = mi48.get_status() & DATA_READY
    # read the frame
    # assert the spi_cs, delay a bit then read
    mi48_spi_cs_n.on()
    time.sleep(MI48_SPI_CS_DELAY)
    data, header = mi48.read()
    if data is None:
        logger.critical('NONE data received instead of GFRA')
        mi48.stop(stop_timeout=1.0)
        sys.exit(1)
    # delay a bit, then deassert spi_cs
    time.sleep(MI48_SPI_CS_DELAY)
    mi48_spi_cs_n.off()

    if args.record:
        write_frame(fd_data, data)

    img = data_to_frame(data, mi48.fpa_shape)
    #
    if header is not None:
        logger.debug('  '.join([format_header(header),
                                format_framestats(data)]))
    else:
        logger.debug(format_framestats(data))

    img8u = cv.normalize(img.astype('uint8'), None, 255, 0,
                         norm_type=cv.NORM_MINMAX,
                         dtype=cv.CV_8U)
    img8u = cv_filter(img8u, parameters={'blur_ks':3}, use_median=False, use_bilat=True, use_nlm=False)
    if GUI:
        cv_display(img8u)
        key = cv.waitKey(1)  # & 0xFF
        if key == ord("q"):
            break
#    time.sleep(1)

# stop capture and quit
mi48.stop(stop_timeout=0.5)
try:
    fd_data.close()
except NameError:
    # file descriptor was closed or never created
    pass
cv.destroyAllWindows()
