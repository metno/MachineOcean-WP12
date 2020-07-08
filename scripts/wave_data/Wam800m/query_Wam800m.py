"""A small example of how to query Wam800m data for the Norwegian coast.

main point of entry in thredds: https://thredds.met.no/thredds/fou-hi/mywavewam800.html
Note: the Wam800m data seems to go only a few years back; check if should use another kind of data input.
Note: these data are available around the different parts of the Norwegian coast, see: https://thredds.met.no/thredds/fou-hi/mywavewam800.html

"""

import netCDF4 as nc4
import motools
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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

nc_lat = np.array(nc_content["latitude"][:][:])
nc_lon = np.array(nc_content["longitude"][:][:])

time = 0

nc_hs = np.array(nc_content["hs"][time][:][:])
# take care of fill values; -999.f, in practise use threshold higher to avoid rounding errors etc
threshold_filling = -500
nc_hs[np.where(nc_hs < threshold_filling)] = np.NaN

fig, ax = plt.subplots()

plt.pcolor(nc_lon, nc_lat, nc_hs)

fig, ax = plt.subplots()

basemap_obj = Basemap(llcrnrlon=5.0, llcrnrlat=57.0, urcrnrlon=12.0, urcrnrlat=60.5,
                      lat_0="58.0", lon_0=7.0,
                      projection="lcc",
                      resolution="i"  # c (crude), l (low), i (intermediate), h (high), f (full)
                      )

basemap_obj.drawcountries()

basemap_obj.drawcoastlines()

# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
parallels = np.arange(50., 75., 2.5)
# labels = [left,right,top,bottom]
basemap_obj.drawparallels(parallels, labels=[False, True, True, False])
meridians = np.arange(-10., 40, 2.5)
basemap_obj.drawmeridians(meridians, labels=[True, False, False, True])

(nc_lon_m, nc_lat_m) = basemap_obj(nc_lon, nc_lat)

my_cmap = plt.get_cmap('rainbow')
cs = basemap_obj.pcolormesh(nc_lon_m, nc_lat_m, nc_hs, cmap=my_cmap, vmin=0.0, vmax=3.0)
basemap_obj.colorbar(cs, extend='min', label="Hs [m]")

plt.tight_layout()
plt.savefig("hs_from_WAM800.pdf")

plt.show()
