# -*- coding: utf-8 -*-
"""Machine Ocean Config Class

 Config class for the Machine Ocean Toolbox
"""

import logging

from os import path

logger = logging.getLogger(__name__)

class Config():

    def __init__(self, version="1.0"):

        self._confFile = None
        self._confVers = version

        self._loadConfig()

        return

    ##
    #  Internal Functions
    ##

    def _loadConfig(self):
        """Load the config file, if it exists, and extract the settings
        for the request config version.
        """
        return True

# END Class Config
