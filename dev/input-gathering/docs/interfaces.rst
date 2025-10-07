.. index:: interfaces

.. py:module:: senxor.interfaces

MI48 Interfaces
====================

The ``senxor.interfaces`` module provides wrappers over file-like
instances of the corresponding physical interfaces.

The wrappers provide three functions to the MI48 class:

* Read a control/status register
* Write a value to a control register
* Read a thermogram

USB Interface
-------------

The USB interface handles both control/status access, as well as
thermogram readout.

.. autoclass:: USB_Interface
   :members:

SPI/I2C Interface
-----------------

If SPI and I2C are used, then thermogram and control/status streams
are separated

.. autoclass:: I2C_Interface
   :members:

.. autoclass:: SPI_Interface
   :members:

