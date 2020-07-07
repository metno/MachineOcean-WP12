"""A small example of how to query Wam800m data for the Norwegian coast.

main point of entry in thredds: https://thredds.met.no/thredds/fou-hi/mywavewam800.html
Note: the Wam800m data seems to go only a few years back

"""

import netCDF4 as nc4
import motools
import subprocess
import os

# TODO: 1) try to download 2) read the downloaded file [note: check first if the file already exists]
# NOTE: this is available over different regions: see https://thredds.met.no/thredds/fou-hi/mywavewam800.html
year = 2019
month = 4
day = 5
forecast_time = 18

filename = "https://thredds.met.no/thredds/fileServer/fou-hi/mywavewam800shf/mywavewam800_skagerak.an.{}{:02}{:02}{:02}.nc".format(year, month, day, forecast_time)
path = "./{}".format(filename[65:])

if not os.path.isfile(path):
    bash_command = "wget {}".format(filename)
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

nc_content = nc4.Dataset(path, "r")


