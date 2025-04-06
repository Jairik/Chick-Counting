PySenXor Documentation
====================================

PySenXor is a Python library for interacting with Meridian Innovation’s camera
modules and thermal imaging processors.
PySenXor aims to bring you a great thermal imaging stream out of the box
with a few lines of code, and enable you to focus on the processing
and analytics of the thermal data for the application you're trying to
enhance by adding thermal imaging.

PySenXor is run on the host system, assuming one of the setups shown in :numref:`system_usb`.
Therefore, PySenXor represents an abstraction over the MI48xx thermal
imaging processor (TIP), regardless of the physical interface between the host
system and the MI48xx (USB or SPI/I2C are currently available).

.. _system_usb:

.. figure:: ./figures/MI48-USB--concept-system.png
   :width: 600

.. figure:: ./figures/MI48-SPI--concept-system.png
   :width: 600

   Conceptual diagrams of two thermal imaging solutions based on Meridian
   Innovation’s camera module and the companion MI48XX IC.


Additionally, it provides:

* blueprints of interaction with the MI48xx which can easily be ported to another language, e.g. C or Java,
* utilities for visualisation of the thermal data via OpenCV,
* examples of optimised temporal and spacial filtering of the thermal data (which inevitably contains noise).

The structure of PySenXor is shown in :numref:`pysenxor_structure`
The main classes are the MI48, the three interface classes USB, SPI,
and I2C.

.. _pysenxor_structure:

.. figure:: ./figures/pysenxor-architecture.png

   Architectural diagrams of the MI48xx TIP, PySenXor, and an
   application built around PySenXor.


In an application, depending on the host interface, one must provide
either a USB_Interface instance, or the pair of SPI_Interface and
I2C_Interface instances, as illustrated in :numref:`pysenxor_structure`.
These interface instances are effectively wrappers around any library
that could provide a file-like access to the  corresponding physical interface, e.g.

* PySerial, for USB interface,
* smbus2 (or smbus) for I2C interface, and
* spidev for SPI interface.

Note that the two major modules of PySenxor -- the mi48.py and the
interfaces.py do not depend on any of the above, and not even on
OpenCV.
The examples however do depend, and they are your starting point.

Enjoy!

Contents
========
.. toctree::
   :maxdepth: 2

   mi48
   interfaces
   utils
   install
   usage


Indices and tables
==================

  + :ref:`genindex`
  + :ref:`modindex`
  + :ref:`search`

