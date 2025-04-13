''' This file aims to capture both the RGB and Thermal Camera video outputs at once, time stamping and saving to a common folder '''

# RGB Camera Libraries
from libcamera import controls
from picamera2 import Picamera2 
# Thermal Camera Libraries
import sys
sys.path.append("/home/test/myenv/lib/python3.11/site-packages")
import os
import signal
from smbus import SMBus
from spidev import SpiDev
import argparse  # Expandability
try:
    from gpiozero import Pin, DigitalInputDevice, DigitalOutputDevice
except:
    print("ERROR- Must install gpiozero with pip3 install gpiozero")
    sys.exit()
import time
from datetime import datetime
import logging
import numpy as np # Plotting thermal values into pictures
import cv2 as cv
from senxor.mi48 import MI48, DATA_READY, format_header, format_framestats
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface

# Configuring logger for debugging purposes
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

# Configuring argument parsing for easy testing and expadability
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record', default=False, dest='record', action='store_true', help='Record data')
    parser.add_argument('-fps', '--framerate', default=15, type=float, help='Desired Framerate', dest='fps')
    args = parser.parse_args()
    return args

print("Dependencies are initialized and set up...")

# Defining helper functions
def get_filename(tag):
    '''Yield a timestamp filename with a specified tag.'''
    now = datetime
    ts = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + f'-{now.microsecond // 1000:03d}'   
    filename = "{}--{}".format(tag, ts)
    return filename

def write_frame(outfile, arr):
    '''Write a numpy array as a row in 
    a file, using C ordering.'''
    if arr.dtype == np.uint16:
        outstr = ('{:n} '*arr.size).format(*arr.ravel(order='C')) + '\n'
    else:
        outstr = ('{:.2f} '*arr.size).format(*arr.ravel(order='C')) + '\n'
    try:
        outfile.write(outstr)
        outfile.flush()  # Flush the buffer since the str is fairly large
    except AttributeError:
        # Manually write to the file line by line
        with open(outfile, 'a') as fh:
            fh.write(outstr)
        return None
    except IOError:
        logger.critical('Cannot write to {} (IOERROR)'.format(outfile))
        sys.exit(106)  # Exit with specific code
        
def cv_display(img, title='', resize=(320, 248)):
    colormap=cv.COLORMAP_JET, interpolation=cv.INTER_CUBIC
    ''' Convert 2D numpy array to image and save to a file '''
    cvcol = cv.applyColorMap(img, colormap)
    cvresize =  cv.resize(cvcol, resize, interpolation=interpolation)
    # Save to folder
    
# Parse the command line arguments    
args = parse_args()

RPI_GPIO_I2C_CHANNEL = 1  # I2C channels available (ls /dev/*i2c* displays devices)
RPI_GPIO_SPI_BUS = 0  # SPI Bus available (ls /dev/*spi* displays devices)
RPI_GPIO_SPI_MI48 = 0  # MI48A routing 
MI48_I2C_ADDRESS = 0x40  # MI48 I2C Address ($i2cdetect -y 1)

# MI48 SPI Stuff:
MI48_SPI_MODE = 0b00
MI48_SPI_BITS_PER_WORD = 8  # Works best with default 8
MI48_SPI_LSBFIRST = False
MI48_SPI_CSHIGH = True
MI48_SPI_MAX_SPEED_HZ = 31200000
MI48_SPI_CS_DELAY = 0.0001  # Delay between asserting/deasserting CS_N and initiating/stopping clock/data

# Once the script starts, safely close the interfaces
def close_all_interfaces():
    try:
        spi.close()
    except NameError:
        pass
    try:
        i2c.close()
    except NameError:
        pass
    
# Define a signal handler to ensure clean closer upon CTRL+C
def signal_handler(sig, frame):
    '''ENsure clean exit in case of SIGINT or SIGTERM'''
    logger.info("Exiting due to SIGINT or SIGTERM")
    mi48.stop(poll_timeout=0.25, stop_timeout=1.2)  # Close all connections from device
    time.sleep(0.5)
    logger.info("Done")
    sys.exit(0)  # Exit with success

# Define the signals that should be handled to ensure clean exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create an I2C interface object
i2c = I2C_Interface(SMBus(RPI_GPIO_I2C_CHANNEL), MI48_I2C_ADDRESS)
