# -*- coding: utf-8 -*-
"""Machine Ocean Tools Init

 Init File for the Machine Ocean Toolbox
"""

import logging
import os
import time

from .config import Config
from .sentineldata import SentinelData

__package__   = "Machine Ocean Tools"
__author__    = "MET Norway"
__copyright__ = "Copyright 2020, MET Norway"
__license__   = ""
__version__   = ""
__url__       = "https://machineocean.met.no"
__credits__   = [
    "Jean Rabault",
    "Veronica Berglyd Olsen",
    "Martin Lilleeng Sætra",
]

__all__ = [
    "Config", "SentinelData"
]

# Initiating logging
strLevel = os.environ.get("MOTOOLS_LOGLEVEL", "INFO")
if hasattr(logging, strLevel):
    logLevel = getattr(logging, strLevel)
else:
    print("Invalid logging level '%s' in environment variable MOTOOLS_LOGLEVEL" % strLevel)
    logLevel = logging.INFO

if logLevel < logging.INFO:
    logFormat = "[{asctime:s}] {levelname:8s} {message:}"
else:
    logFormat = "{levelname:8s} {message:}"

logging.basicConfig(format=logFormat, style="{", level=logLevel)
logger = logging.getLogger(__name__)

# Make sure the interpreter is in UTC in all the following
os.environ["TZ"] = "UTC"
time.tzset()
