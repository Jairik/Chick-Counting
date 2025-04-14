# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019. All rights reserved.
#
import os
import time
import numpy as np
import logging
import serial

try:
    import matplotlib
    from matplotlib import pyplot as plt
except:
    print("Please install matplotlib to see the thermal image")

from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import data_to_frame, connect_senxor

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

mi48, connected_port, port_names = connect_senxor()

# print out camera info
logger.info('Camera info:')
logger.info(mi48.camera_info)

# initiate single frame acquisition
with_header = True
mi48.start(stream=False, with_header=with_header)

# Read the frame
data, header = mi48.read()
# Log the header and frame stats
if header is not None:
    logger.debug('  '.join([format_header(header),
                            format_framestats(data)]))
else:
    logger.debug(format_framestats(data))

# Visualise data after reshaping the array properly
img = data_to_frame(data, mi48.fpa_shape)
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
