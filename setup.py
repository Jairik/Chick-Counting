#!/usr/bin/env python3
#
# Copyright (C) 2019-2022  Stanislav Markov, Meridian Innovation Ltd, Hong Kong

from setuptools import setup
import sys
import os

if sys.version_info < (3, 6, 0, 'final', 0):
    raise SystemExit('Python 3.6 is required!')

_install_requires = [
    # core
    'pyserial',
    'smbus; platform_system=="Linux"',
    'spidev; platform_system=="Linux"',
    "numpy",
    "crcmod",
    "opencv-python",
    #
    # to use RGB camera in addition to the visual
    "imutils",
    #
    # to use plots, histograms, colormap etc from matplotlib
    "matplotlib",
    "cmapy"
]
try:
    # add gpiozero to raspberry pi system; but below may fail on other arm host
    if os.uname()[4].startswith("arm"):
        _install_requires.append('gpiozero')
except AttributeError:
    # windows does not have os.uname(); one can confirm we're on windows by
    # os.name == 'nt', but we do not really care
    pass

setup(
    name="pysenxor",
    version="1.4.1",
    description="Python SDK for Meridian Innovation's SenXor.",
    author="Stanislav Markov, Meridian Innovation Ltd.",
    platforms="platform independent",
    package_dir={"": "."},
    packages=[
        "senxor",
    ],
    scripts=[],
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    long_description="""

""",
    install_requires = _install_requires,
)
