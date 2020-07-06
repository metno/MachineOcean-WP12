"""Script to plot the Norwegian storm surge stations locations from the data in the
kyststasjoner_norge netCDF files. """

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc4

fig, ax = plt.subplots()

# setup Lambert Conformal basemap.
basemap_obj = Basemap(llcrnrlon=-8.0, llcrnrlat=55.5, urcrnrlon=34.5, urcrnrlat=72.0,
                      lat_0="65.0", lon_0=15.0,
                      projection="lcc",
                      resolution="l"  # c (crude), l (low), i (intermediate), h (high), f (full)
                      )

basemap_obj.drawcountries()

basemap_obj.drawcoastlines()

# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
basemap_obj.drawmapboundary(fill_color="#A6CAE0")

# fill continents, set lake color.
basemap_obj.fillcontinents(color='grey', lake_color='lavender')

# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
parallels = np.arange(50., 75., 10.)
# labels = [left,right,top,bottom]
basemap_obj.drawparallels(parallels, labels=[False, True, True, False])
meridians = np.arange(-10., 40, 10.)
basemap_obj.drawmeridians(meridians, labels=[True, False, False, True])

# get the data from the example file
folder = "../../example_data/StormSurge/NorwegianStations/"
path = folder + "kyststasjoner_norge.nc2019110700"

nc_content = nc4.Dataset(path, 'r')

lat_data = nc_content["latitude"][:][0]
lon_data = nc_content["longitude"][:][0]

for ind, (crrt_lat, crrt_lon) in enumerate(zip(lat_data, lon_data)):
    plot_lon, plot_lat = basemap_obj(crrt_lon, crrt_lat)
    basemap_obj.scatter(plot_lon, plot_lat, marker='o', color='r', zorder=5)
    ax.annotate(str(ind), (plot_lon, plot_lat),
                xytext=(5, 5), textcoords="offset points", color="red")

plt.tight_layout()
plt.savefig("storm_surge_positions.pdf")
plt.show()
