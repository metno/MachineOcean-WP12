# -*- coding: utf-8 -*-
"""Machine Ocean Config Class

 Config class for the Machine Ocean Toolbox
"""

import sys
import json
import logging

from os import path

logger = logging.getLogger(__name__)

class Config():

    def __init__(self, version="1.0"):

        self._packRoot = None
        self._confFile = None
        self._confVers = version
        self._confData = {}

        self._loadConfig()
        # print(self._confData)

        return

    ##
    #  Internal Functions
    ##

    def _loadConfig(self):
        """Load the config file, if it exists, and extract the settings
        for the request config version.
        """
        self._packRoot = getattr(sys, "_MEIPASS", path.abspath(path.dirname(__file__)))
        confDir = path.abspath(path.join(self._packRoot, path.pardir, "config"))
        confFile = path.join(confDir, "config.json")

        if not path.isdir(confDir):
            logger.error("Config folder is missing. It should be at: %s" % confDir)
        if not path.isfile(confFile):
            logger.error("Config file is missing. It should be at: %s" % confFile)

        jsonData = {}
        try:
            with open(confFile, mode="r") as inFile:
                jsonData = json.loads(inFile.read())
        except Exception as e:
            logger.error("Failed to parse config JSON data.")
            logger.error(str(e))
            return False

        if "config" not in jsonData:
            logger.error("Root entry of the config file is not 'config'")
            return False

        if self._confVers not in jsonData["config"]:
            logger.error("Unknown config version '%s'" % self._confVers)
            return False

        self._confData = jsonData["config"][self._confVers]
        logger.debug("Config data successfully loaded from JSON file.")

        return True

# END Class Config
