''' Test program to ping SPI to determine if it is properly working. If it is, will print a response'''

import spidev

spi = spidev.SpiDev()

spi.open(0, 0)  # Bus 0, CE0
spi.mode = 0b00  # SPI Mode 0
spi.bits_per_word = 8

spi.max_speed_hz = 31200000  # Adjust the speed as needed
response = spi.xfer2([0x01, 0x80, 0x00])  # Send a sample command

print(response)  # Will print something if detected
spi.close()

