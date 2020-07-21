# -*- coding: utf-8 -*-
"""Machine Ocean Tools Init

 Init File for the Machine Ocean Toolbox
"""

import logging
import os
import time

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

# Initiating logging
logger = logging.getLogger(__name__)

# Make sure the interpreter is in UTC in all the following
os.environ["TZ"] = "UTC"
time.tzset()
