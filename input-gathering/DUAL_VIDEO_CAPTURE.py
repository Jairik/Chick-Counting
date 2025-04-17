''' This file aims to capture both the RGB and Thermal Camera video outputs at once, time stamping and saving to a common folder '''

# RGB Camera Libraries
from libcamera import controls
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
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
    parser.add_argument('-thermalfps', '--thermalframerate', default=25, type=float, help='Desired Framerate for Thermal Camera', dest='thermalframerate')
    parser.add_argument('-rgbfps', '--rgbframerate', default=30, type=float, help='Desired Framerate for RGB Camera', dest='fps')
    parser.add_argument('-rgbpreview', '--rgbvideopreview', default=False, type=bool, help='See a preview of the RGB Camera')
    parser.add_argument('-thermalpreview', '--thermalcamerapreview', default=False, type=bool, help='See a preview of the Thermal Camera')
    args = parser.parse_args()
    return args

print("Dependencies are initialized and set up...")

''' Setting up & Calibrating the Thermal Camera '''

# Defining helper functions
def get_filename(tag, cameraType="thermal"):
    '''Yield a timestamp filename with a specified tag.'''
    now = datetime.now()
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
    #Ensure clean exit in case of SIGINT or SIGTERM
    logger.info("Exiting due to SIGINT or SIGTERM")
    mi48.stop(poll_timeout=0.5, stop_timeout=1.2)  # Close all connections from device
    time.sleep(0.5)
    logger.info("Done")
    sys.exit(0)  # Exit with success'


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
rgb_config = picam2.create_video_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},
    encode="main"  # Name of steam to encode
)
picam2.configure(rgb_config)  # Forcing correct resolution
picam2.start()
encoder = H264Encoder(bitrate=10_000_000)
print("RGB Camera Initialized")
w, h, rgb_fps = 1920, 1080, 30
#w, h, fps = 1280, 720, 60  # Utilized if higher framerate is needed

tw, th = 62, 80  # Setting thermal camera dimensions (known)

# Defining a video writer for rgb & thermal cameras (to save video to file)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rgb_filename = get_filename(tag='', cameraType="RGB") + ".m264"
thermal_filename = get_filename(tag='', cameraType="Thermal") + ".mp4"
# rgb_output = cv2.VideoWriter(rgb_filename, fourcc, rgb_fps, (w, h))
thermal_output = cv2.VideoWriter(thermal_filename, fourcc, args.thermalframerate, (tw, th))
# Defining RGB dimensions based off dynamic test
# initial_frame = picam2.capture_array()
# actual_h, actual_w = initial_frame.shape[:2]
# logger.debug(f"Captured frame dimensions for RGB Camera: width={actual_w}, height={actual_h}, expected: w={w}, h={h}")
#rgb_output = cv2.VideoWriter(rgb_filename, fourcc, rgb_fps, (actual_w, actual_h))  # Dynamically setting video writer with actual dimensions
# Ensuring that the video writers are successfully opened
#if not rgb_output.isOpened():
#    logger.error(f"Could not open RGB Writer (filename={rgb_filename})")
#    sys.exit(1)

print("Video Writers Initialized")

# Defining a global scope event
stop_event = threading.Event()

# Defining a signal handler so everything successfully exits on Ctrl+C
def safe_exit_signal_handler(sig, frame):
    logger.info("Exiting due to SIGINT or SIGTERM")
    stop_event.set()  # Signal all threads to stop

    time.sleep(0.5)  # Wait a small amount of time to threads to finish loops

    try:
        mi48.stop(poll_timeout=0.5, stop_timeout=1.2)
    except Exception as e:
        logger.error(f"Error stopping MI48: {e}")

    logger.info("Signal Handler is done")
    # Now, we  let main safely exit itself

# Registering the signal handlers
signal.signal(signal.SIGINT, safe_exit_signal_handler)
signal.signal(signal.SIGTERM, safe_exit_signal_handler)

''' RGB Camera Loop to capture video '''
def capture_rgb():
    try:
 #       while not stop_event.is_set():
            # Capture the frame
#            frame = picam2.capture_array()
            
            # Display message if dimensions are off
#            if frame.shape[0] != h or frame.shape[1] != w:
#                logger.critical(f"RGB Camera dimensions are off (actual {h}, {w}")
                # sys.exit(10)  # Exit with exit code

            # Write the frame to the video file
#            rgb_output.write(frame)
            
            # If user chooses, show video output
#            if args.rgbvideopreview: cv2.imshow("Video", frame)
            
            # Break loop on pressing 'q'
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break
        picam2.start_recording(encoder, rgb_filename)
    except KeyboardInterrupt:
        picam2.stop_recording()
        cur_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + f'-{datetime.now().microsecond // 1000:03d}'
        print(f"RGB video capture stopped at {cur_time}")

''' Thermal Camera Loop to capture video & collect data '''
# args.thermalcamerapreview
def capture_thermal():
    try:
        while not stop_event.is_set():
            if hasattr(mi48, 'data_ready'):
                mi48.data_ready.wait_for_active()
            else:
                data_ready = False
                while not data_ready and not stop_event.is_set():
                    time.sleep(0.005)
                    data_ready = mi48.get_status() & DATA_READY
                if stop_event.is_set(): break  # Ensuring frequent check for signal
            # Read the frame
            mi48_spi_cs_n.on()
            # time.sleep(MI48_SPI_CS_DELAY)
            data, header = mi48.read()
            if data is None:
                logger.critical('NONE data received instead of GFRA')
                mi48.stop(stop_timeout=1.0)
                sys.exit(1)  # Exit with error
            # delay a bit, then deassert spi_cs
            time.sleep(MI48_SPI_CS_DELAY)
            mi48_spi_cs_n.off()

            if args.record:
                write_frame(fd_data, data)

            img = data_to_frame(data, mi48.fpa_shape)

            if header is not None:
                logger.debug('  '.join([format_header(header),
                                        format_framestats(data)]))
            else:
                logger.debug(format_framestats(data))

            img8u = cv.normalize(img.astype('uint8'), None, 255, 0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            img8u = cv_filter(img8u, parameters={'blur_ks':3}, use_median=False, use_bilat=True, use_nlm=False)
            
            if args.thermalcamerapreview:
                cv_display(img8u)
                key = cv.waitKey(1)  # & 0xFF
                if key == ord("q"):
                    break
            thermal_colored_frame = cv.applyColorMap(img8u, cv.COLORMAP_JET)
            thermal_resized_frame = cv.resize(thermal_colored_frame, (tw, th))
            thermal_output.write(thermal_resized_frame)
            
    except KeyboardInterrupt:
        cur_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + f'-{datetime.now().microsecond // 1000:03d}'
        print(f"Thermal video capture stopped at {cur_time}")
    finally:
        thermal_output.release()
        print("Thermal Camera Video Writer Released")

''' Creating threads to divide thermal camera and rgb camera '''
rgb_thread = threading.Thread(target=capture_rgb)  # No args needed
thermal_thread = threading.Thread(target=capture_thermal)  # No args needed

# Starting the threads
rgb_thread.start()
thermal_thread.start()

# Wait for both threads to finish cleanup
rgb_thread.join()
thermal_thread.join()

# Allow threads some more time to finish
print("Sleeping for a little to allow threads to complete...")
time.sleep(.5)

''' Release resources/additional cleanup once both threads are finished '''
print("Releasing resources now...")
picam2.stop()
rgb_output.release()
thermal_output.release()
try:
    mi48.stop(stop_timeout=0.5)
except Exception:
    pass  # Sometimes errors out, but is fine
try:
    fd_data.close()
except NameError:
    pass  # File descriptor is already closed
cv2.destroyAllWindows()
