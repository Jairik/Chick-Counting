# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019. All rights reserved.
import sys
import os
from smbus import SMBus
from spidev import SpiDev
import time
import logging
import numpy as np
from gpiozero import Pin, DigitalInputDevice, DigitalOutputDevice

try:
    import matplotlib
    from matplotlib import pyplot as plt
except:
    print("Please install matplotlib to see the thermal image")

from pprint import pprint

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

from senxor.mi48 import MI48, DATA_READY, format_header, format_framestats
from senxor.utils import data_to_frame
from senxor.interfaces import SPI_Interface, I2C_Interface


def get_single_frame(mi48, with_header=True, read_delay=40, poll_frame_mode=False):
    """Return a frame from a single acquisition."""
    # check status and mode
    mi48.get_mode(verbose=True)
    mi48.get_status(verbose=True)
    # initiate capture
    mi48.start(stream=False, with_header=with_header)
    t0 = time.time()
    # wait for data_ready pin (or poll for STATUS.DATA_READY /fw 2.1.X+)
    if hasattr(mi48, 'data_ready'):
        mi48.data_ready.wait_for_active()
    else:
        data_ready = False
        while not data_ready:
            time.sleep(0.01)
            data_ready = mi48.get_status() & DATA_READY

#    # read the frame
    t1 = time.time()
    print ('Approximate capture time: {:.0f} ms'.format(1.e3 * (t1-t0)))
    # assert the spi_cs, delay a bit then read
    mi48_spi_cs_n.on()
    time.sleep(MI48_SPI_CS_DELAY)

    data, header = mi48.read()

    time.sleep(MI48_SPI_CS_DELAY)
    mi48_spi_cs_n.off()

    mi48.get_mode(verbose=True)
    mi48.get_status(verbose=True)
    return data, header


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
# spidev.bufsiz=<NEEDED BUFFER SIZE>
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
# spi.device.cshigh = MI48_SPI_CSHIGH
spi.device.cshigh = True
spi.device.no_cs = True
mi48_spi_cs_n = DigitalOutputDevice("BCM7", active_high=False,
                                    initial_value=False)


# connect the reset line to allow to drive it by SW (GPIO23, J8:16)
mi48_reset_n = DigitalOutputDevice("BCM23", active_high=False,
                                   initial_value=True)

class MI48_reset:
    def __init__(self, pin,
                 assert_seconds=0.00005,
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


use_data_ready_pin = True
if use_data_ready_pin:
    mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False)

#mi48 = MI48([i2c, spi])
mi48 = MI48([i2c, spi], data_ready=mi48_data_ready,
            reset_handler=MI48_reset(pin=mi48_reset_n))

#mi48.reset()
# print out camera info
camera_info = mi48.get_camera_info()
print('Camera info:')
pprint(camera_info)

# set desired FPS
STREAM_FPS = 9
mi48.set_fps(STREAM_FPS)
# the following sets appropriate read_delay if
# capture_in_progress is not polled for FW 1.x.x
max_fps = 25.5
div = mi48.get_frame_rate()
read_delay = 1.e3 * ( div / max_fps )

# print out control/status registers
mi48_regs = mi48.get_ctrl_stat_regs()
print('\nMI48 registers:')
pprint(mi48_regs)

# get a single capture
crc_error = True
nbad = 0

while crc_error:
    data, header = get_single_frame(mi48, with_header=True,
                                    poll_frame_mode=False,
                                    read_delay=read_delay)
    if header is not None:
        print("\nFrame Header:")
        pprint(header)
    if mi48.crc_error:
        nbad += 1
        print('Bad frames: {}'.format(nbad))
        print(data[:20])
    else:
        crc_error = False
    if nbad == 2: raise RuntimeError

#
nogui = False
if len(sys.argv) == 2 and sys.argv[1]=='-nogui': nogui = True

# Visualise data after reshaping the array properly
img = data_to_frame(data, mi48.fpa_shape)
print("\nFrame stats:")
print("Min: {:.1f}, Max: {:.1f}, Avg: {:.1f}, Stdev: {:.1f}".
      format(img.min(), img.max(), img.mean(), img.astype(np.float32).std()))

if not nogui:
    try:
        img = plt.imshow(img.astype(np.float32), cmap='coolwarm',
                        aspect='equal', interpolation=None)
        plt.axis('off')
        plt.show()
    except NameError:
        # plt not found/not imported/missing
        pass

# stop capture and quit
mi48.stop()
