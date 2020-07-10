"""A small example of how to query the MEPS data for the thredds server.

main point of entry is: https://thredds.met.no/thredds/catalog/meps25epsarchive/catalog.html
NOTE: data seem to go only a few years back; check for availability of older data.

"""

import netCDF4 as nc4
import motools.helper.arrays as moa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

year = 2019
month = 4
day = 5
forecast_time = 18

url = "https://thredds.met.no/thredds/dodsC/meps25epsarchive/{}/{:02}/{:02}/meps_subset_2_5km_{}{:02}{:02}T{:02}Z.nc".format(year, month, day, year, month, day, forecast_time)

print("connect to OpenDAP server")
nc_dataset = nc4.Dataset(url)

print("retrieve latitude and longitude grid data")
nc_lat = np.array(nc_dataset["latitude"][:][:])
nc_lon = np.array(nc_dataset["longitude"][:][:])

print("compute slices corresponding to the domain of interest")
range_of_interest_lat = [56.5, 60.5]
range_of_interest_lon = [6.0, 11.0]
indexes_of_interest = moa.index_ranges_within_bounds(nc_lon, nc_lat, range_of_interest_lon, range_of_interest_lat)
(min_index_0, max_index_0, min_index_1, max_index_1) = indexes_of_interest
nc_lon_red = nc_lon[min_index_0:max_index_0, min_index_1:max_index_1]
nc_lat_red = nc_lat[min_index_0:max_index_0, min_index_1:max_index_1]

# TODO: request only the part of the data needed
print("retrieve surface air pressure data")
time = 0
height_0 = 0
ensemble_member = 0

# both these syntaxes should be equivalent, and actually retrieve the full "surface_air_pressure" data field before slicing
# nc_data_sea_pressure = np.array(nc_dataset["surface_air_pressure"][time][height_0][ensemble_member][:][:])
# nc_data_sea_pressure = nc_data_sea_pressure[min_index_0:max_index_0, min_index_1:max_index_1]
# equivalent:
# nc_data_sea_pressure = np.array(nc_dataset["surface_air_pressure"][ time, height_0, ensemble_member, min_index_0:max_index_0, min_index_1:max_index_1])

# it looks like, to retrive only the indexes wanted, one has to formulate the url by hand; I cannot find another method from the doc / resources online
# url requests do not follow python conventions; this retrieves [min_index;max_index], max_index is INCLUDED
url_request = url + "?surface_air_pressure[0][0][0][{}:1:{}][{}:1:{}]".format(min_index_0, max_index_0-1, min_index_1, max_index_1-1)
nc_data_sea_pressure = np.array(nc4.Dataset(url_request)["surface_air_pressure"][time][height_0][ensemble_member][:][:])

print("show the data")
plt_downsample = 1

fig, ax = plt.subplots()

basemap_obj = Basemap(llcrnrlon=4.0, llcrnrlat=55.5, urcrnrlon=13.0, urcrnrlat=61.5,
                      lat_0="58.0", lon_0=7.0,
                      projection="lcc",
                      resolution="l"  # c (crude), l (low), i (intermediate), h (high), f (full)
                      )

basemap_obj.drawcountries()

basemap_obj.drawcoastlines()

# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
# parallels = np.arange(50., 75., 2.5)
parallels = np.array([56.5, 60.5])
basemap_obj.drawparallels(parallels, labels=[False, True, True, False])
# meridians = np.arange(-10., 40, 2.5)
meridians = np.array([6.0, 11.0])
basemap_obj.drawmeridians(meridians, labels=[True, False, False, True])

(nc_lon_m, nc_lat_m) = basemap_obj(nc_lon_red, nc_lat_red)

my_cmap = plt.get_cmap('rainbow')
cs = basemap_obj.pcolormesh(nc_lon_m[::plt_downsample], nc_lat_m[::plt_downsample], nc_data_sea_pressure[::plt_downsample], cmap=my_cmap)
cb = basemap_obj.colorbar(cs, extend='min', label="sea level pressure", pad=0.50)

plt.tight_layout()

plt.savefig("air_pressure_at_sea_level.pdf")

plt.show()
