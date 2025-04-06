.. index:: install

Installation
============

Download the repo or unpack the tarball.

Execute the following from the directory of the pysenxor package:

* If in a virtual environtment (RECOMMENDED)

.. code:: bash

   pip3 install -e .


* If NOT in a virtual environtment (NOT recommended)

.. code:: bash

   pip3 install --user -e .


Dependencies (handled automatically by PIP):

* pyserial (usb)
* smbus (i2c)
* spidev (spi)
* crcmod (CRC calculations)
* opencv-python (video, image analytics)
* matplotlib (plenty of colormaps, figures, etc)
* cmapy (colormaps from matplotlib)
* imutils (access to RGB webcam and some transforms)
