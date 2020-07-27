# -*- coding: utf-8 -*-
"""Machine Ocean Config Class

 Config class for the Machine Ocean Toolbox
"""

import sys
import json
import logging

from os import path

logger = logging.getLogger("motools")

class Config():

    def __init__(self):

        self._packRoot = None
        self._confData = {}

        self._loadConfig()
        print(self._confData)

        return

    ##
    #  Internal Functions
    ##

    def _loadConfig(self):
        """Load the config files, if they exist, and extract the data.
        """
        self._packRoot = getattr(sys, "_MEIPASS", path.abspath(path.dirname(__file__)))
        rootDir = path.abspath(path.join(self._packRoot, path.pardir))
        logger.debug("MOTools root dir is: %s" % rootDir)

        metConf = path.join(rootDir, "met_config", "met_config.json")
        mainConf = path.join(rootDir, "main_config.json")
        userConf = path.join(rootDir, "user_config.json")

        self._confData = {
            "MET":  {"path": metConf,  "config": {}, "loaded": False},
            "MAIN": {"path": mainConf, "config": {}, "loaded": False},
            "USER": {"path": userConf, "config": {}, "loaded": False},
        }

        for confGroup in self._confData:
            confFile = self._confData[confGroup]["path"]
            logger.debug("Loading %s config file" % confGroup)
            if path.isfile(confFile):
                jsonData = {}
                try:
                    with open(confFile, mode="r") as inFile:
                        jsonData = json.loads(inFile.read())
                    if "config" in jsonData:
                        self._confData[confGroup]["config"] = jsonData["config"]
                        self._confData[confGroup]["loaded"] = True
                except Exception as e:
                    logger.error("Failed to parse config JSON data.")
                    logger.error(str(e))
                    return False
            else:
                logger.debug("No file: %s" % confFile)

        # if not self._confData["MAIN"]["loaded"]:
        #     logger.error("Failed to load minimum configuration file main_config.json.")
        #     raise RuntimeError

        return

# END Class Config
