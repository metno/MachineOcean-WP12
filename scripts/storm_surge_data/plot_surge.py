"""Script to plot the storm surge model prediction vs actual measurements"""

import datetime
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt

import motools

# get the data from the example file
folder = "../../example_data/StormSurge/NorwegianStations/"
path = folder + "kyststasjoner_norge.nc2019110700"

nc_content = nc4.Dataset(path, 'r')

nc_time = nc_content["time"][:]
nc_datetime = [datetime.datetime.fromtimestamp(crrt_time) for crrt_time in nc_time]

datafield_model = "totalwater"
datafield_stations = "observed"
datafield_tide = "tide"
nbr_ensemble = 52
nbr_stations = 23

nc_water_model = nc_content[datafield_model][:]
nc_water_station = nc_content[datafield_stations][:]
nc_water_tide = nc_content[datafield_tide][:]
nc_water_station_notide = nc_water_station - nc_water_tide

nc_water_model_mean = np.mean(nc_water_model, axis=1)
nc_water_model_mean_notide = nc_water_model_mean - nc_water_tide
nc_water_model_std = np.std(nc_water_model, axis=1)

nc_water_model_error = nc_water_station - nc_water_model_mean

fig, ax = plt.subplots()

show_ensemble_simulations = False
if show_ensemble_simulations:
    for ensemble_nbr in range(nbr_ensemble-1):
        plt.plot(nc_datetime, nc_water_model[:, ensemble_nbr, 0, 0] - nc_water_tide[:,  0, 0],
                color='k', linewidth=0.5)
        plt.plot(nc_datetime, nc_water_model[:, ensemble_nbr, 0, 0] - nc_water_tide[:,  0, 0],
                 color='k', linewidth=0.5,
                 label=datafield_model)

plt.plot(nc_datetime, nc_water_model_mean_notide[:, 0, 0],
         label="prediction - tide, mean, 3 std".format(datafield_model),
         linewidth=2.5, color="blue")

ax.fill_between(nc_datetime,
                nc_water_model_mean_notide[:, 0, 0] - 3 * nc_water_model_std[:, 0, 0],
                nc_water_model_mean_notide[:, 0, 0] + 3* nc_water_model_std[:, 0, 0],
                alpha=0.2,
                color="blue")

plt.plot(nc_datetime, nc_water_station_notide[:,  0, 0],
         label="{} - tide".format(datafield_stations), color='r',
         linewidth=2.5)

plt.legend()

fig, ax = plt.subplots()

for station in range(nbr_stations):
    plt.plot(nc_datetime, nc_water_model_error[:, 0, station],
            label="{}".format(station))

plt.legend()
plt.ylabel("model error")

plt.show()

"""
- start by thinking in terms of mean
- note that this may also need some distribution approaches
- direct ML approach
- use ML to predict some forms of correlation etc
- problem: how to get ML to become good on 'extreme' events?
- get data from the other stations (UK, France, Sweden, Danmark, etc...) as predictors
- get data from AROME as predictors
- get data from Feroe islands
- get data from Northen Isles over UK
"""
