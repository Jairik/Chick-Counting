''' This file aims to capture both the RGB and Thermal Camera video outputs at once, time stamping and saving to a common folder '''

# RGB Camera Libraries
from libcamera import controls
from picamera2 import Picamera2 
import cv2
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

import threading  # For multithreading with RGB and Thermal Outputs

# Configuring logger for debugging purposes
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

# Configuring argument parsing (Thermal Camera) for easy testing and expadability
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record', default=True, dest='record', action='store_true', help='Record data')
    parser.add_argument('-thermalfps', '--thermalframerate', default=15, type=float, help='Desired Framerate for Thermal Camera', dest='fps')
    parser.add_argument('-rgbfps', '--rgbframerate', default=30, type=float, help='Desired Framerate for RGB Camera', dest='fps')
    args = parser.parse_args()
    return args

print("Dependencies are initialized and set up...")

''' Setting up & Calibrating the Thermal Camera '''

# Defining helper functions
def get_filename(tag, cameraType="thermal"):
    '''Yield a timestamp filename with a specified tag.'''
    now = datetime
    ts = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + f'-{now.microsecond // 1000:03d}'   
    filename = "{}-{}--{}".format(cameraType, tag, ts)
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
RPI_GPIO_SPI_CE_MI48 = 0
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
    '''Ensure clean exit in case of SIGINT or SIGTERM'''
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

# Creating a SPI interface object
SPI_XFER_SIZE_BYTES = 160  # bytes
spi = SPI_Interface(SpiDev(RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48),
                    xfer_size=SPI_XFER_SIZE_BYTES)

spi.device.mode = MI48_SPI_MODE
spi.device.max_speed_hz = MI48_SPI_MAX_SPEED_HZ
spi.device.bits_per_word = 8
spi.device.lsbfirst = False
spi.cshigh = True
spi.no_cs = True
mi48_spi_cs_n = DigitalOutputDevice("BCM7", active_high=False,
                                    initial_value=False)

# Configure DATA_READY and RESET
use_data_ready_pin = True
if use_data_ready_pin: mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False)

# Connect to reset line to allow GPIO23 to drive it
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

# Creating a MI48 Object
mi48 = MI48([i2c, spi], data_ready=mi48_data_ready,
            reset_handler=MI48_reset(pin=mi48_reset_n))

# Printing out the camera info for easy logging
camera_info = mi48.get_camera_info()
logger.info('Camera info:')
logger.info(camera_info)

# Set the desired FPS
mi48.set_fps(args.fps)

# Calibrating the MI48
if int(mi48.fw_version[0]) >= 2:
    mi48.enable_filter(f1=True, f2=True, f3=False) # Enable filtering with default strengths
    mi48.set_offset_corr(0.0)

# Initiaite continious frame aquisition
with_header = True

# Enable saving to a file
if args.record:
    filename = get_filename(mi48.camera_id_hexsn)
    fd_data = open(os.path.join('.', filename+'dat'), 'w')
    
# Starting the thermal camera
mi48.start(stream=True, with_header=with_header)
print("Thermal Camera is started")

''' Setting up & calibrating the RGB Camera '''

# Ensure that camera is available
camera_info = Picamera2.global_camera_info()
if not camera_info:
    print("No cameras detected. Check camera connection and configuration")
    exit(1)
    
# Proceed with camera initialization
picam2 = Picamera2()
picam2.start()
print("Camera Initialized")
w, h, fps = 1920, 1080, 30
#w, h, fps = 1280, 720, 60  # Utilized if higher framerate is needed

# Defining a video writer (to save video to file)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
rgb_filename = get_filename(tag='', cameraType="RGB")
output = cv2.VideoWriter(rgb_filename, fourcc, fps, (w, h))

''' RGB Camera Loop to capture video '''
def capture_rgb(name):
    pass  # Logic Here

''' Thermal Camera Loop to capture video & collect data '''
def capture_thermal(name):
    pass  # Logic Here

''' Creating threads to divide thermal camera and rgb camera '''
rgb_thread = threading.Thread(target=capture_rgb)  # No args needed
thermal_thread = threading.Thread(target=capture_thermal)  # No args needed

# Starting the threads
rgb_thread.start()
thermal_thread.start()

# Wait for both threads to finish 
rgb_thread.join()
thermal_thread.join()

''' Release resources once both threads are finished '''
picam2.stop()
output.release()
mi48.stop(stop_timeout=0.5)
try:
    fd_data.close()
except NameError:
    pass  # File descriptor is already closed
cv2.destroyAllWindows()