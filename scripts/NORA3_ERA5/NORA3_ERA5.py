"""NORA3 and ERA5 tools. 

Install requirements with pip3 install -r requirements.txt
"""

#%%
import netCDF4 as nc4
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_timeseries(param, lat, lon, start_time, end_time, use_atm=True):
    """Time series extraction from NORA3 and ERA5.

    WARNING: Only ERA5 implemented.

    Returns tuple with string "NORA3" or "ERA5" depending on which data set was
    used, and time series xarray for parameter param at location (lat, lon) in 
    interval [start_time, end_time] with a temporal resolution of one hour. If 
    NORA3 data is not available for the given location, ERA5 data will be used. 
    The nearest grid point will be used. Returns an empty xarray if time 
    interval is outside the intervals listed under "Data directories".

    Data directories: 
    /lustre/storeB/project/fou/om/ERA/ERA5 [1979-1, 2019-12]
    /lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf [1997-08, 2019-12]
    """
    nora3_da = get_nora3_timeseries(param, lat, lon, start_time, end_time, use_atm)
    if nora3_da.time.size == 0:
        return ("ERA5", get_era5_timeseries(param, lat, lon, start_time, end_time, use_atm))
    else:
        return ("NORA3", nora3_da)

def get_era5_timeseries(param, lat, lon, start_time, end_time, use_atm=True):
    """Time series extraction from ERA5.

    Returns time series xarray for parameter param at location (lat, lon) in interval
    [start_time, end_time] with a temporal resolution of one hour. The nearest 
    grid point will be used. Returns an empty xarray if time interval is outside
    the intervals listed under "Data directories".

    Data directories: 
    /lustre/storeB/project/fou/om/ERA/ERA5 [1979-1, 2019-12]
    """

    # sanity check arguments
    available_atm_params = ["msl", "u10", "v10"]
    available_wave_params = ["msl", "mwd", "swh"]
    if param not in available_atm_params and param not in available_wave_params:
        raise RuntimeError("Undefined parameter: " + param)
    if -90.0 > lat > 90.0:
        raise RuntimeError("Latitude (lat) must be in the interval [-90.0, 90.0]")
    if 0.0 > lon >= 360.0:
        raise RuntimeError("Longitude (lon) must be in the interval [0.0, 360.0)")
    
    #print("From " + start_time.strftime("%Y%m-%H%M"))
    #print("To " + end_time.strftime("%Y%m-%H%M"))

    # find and open correct netCDF file(s)
    filenames = []
    month_count = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month) + 1
    
    #print("Months:" + str(month_count))

    if param in available_atm_params and use_atm:
        for month in (start_time + relativedelta(months=+n) for n in range(month_count)):
            filenames.append("/lustre/storeB/project/fou/om/ERA/ERA5/atm/era5_atm_CDS_{}.nc".format(month.strftime("%Y%m")))
    elif param in available_wave_params:
        for month in (start_time + relativedelta(months=+n) for n in range(month_count)):
            filenames.append("/lustre/storeB/project/fou/om/ERA/ERA5/wave/era5_wave_CDS_{}.nc".format(month.strftime("%Y%m")))
    else:
        raise RuntimeError(param + " is not found in ERA5 wave data set (try use_atm=True)")
    
    era5 = xr.open_mfdataset(filenames, parallel=True)

    # extract data set
    era5_da = era5[param].sel(longitude=lon, latitude=lat, method="nearest")
    era5_da = era5_da.sel(time=slice(start_time, end_time))

    # return time series as xarray
    return era5_da

def get_nora3_timeseries(param, lat, lon, start_time, end_time, use_atm=True):
    """Time series extraction from NORA3.

    Returns time series xarray for parameter param at location (lat, lon) in interval
    [start_time, end_time] with a temporal resolution of one hour. The nearest 
    grid point will be used. Returns an empty xarray if time interval is outside
    the intervals listed under "Data directories".

    Data directories: 
    /lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf [1997-08, 2019-12]
    """
    return xr.DataArray([], dims={"time": []})

# test time series extraction and plotting from ERA5 data set
ds, da = get_timeseries("swh", 58.0, 10.0, datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 2, 1, 23, 0, 0))
print(ds)
print(da)
print(da.values[0]) # .values is a numpy.ndarray
da.plot()

#%%
def plot_era5_test():
    """Hardcoded experimental function. Do not use! / Will be removed."""
    era5 = xr.open_mfdataset("/lustre/storeB/project/fou/om/ERA/ERA5/atm/era5_atm_CDS_201901.nc")

    print(era5.var)

    era5_da = era5.msl.sel(time="2019-01-01 00:00:00", #expver=5,
        longitude=slice(6.0, 11.0), latitude=slice(60.5, 56.5))

    print(era5_da)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines() 
    era5_da.plot()
    plt.show()

# %%

def plot_nora3_test():
    """Hardcoded experimental function. Do not use! / Will be removed."""
    nora3 = xr.open_mfdataset("/lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf/2019/01/01/00/fc*00*_fp.nc")
    nora3.set_coords(["latitude", "longitude"])

    print(nora3.var)
    print(nora3.projection_lambert)

    # find x, y from lat, lon
    lat = [56.5, 60.5]
    lon = [6.0, 11.0]
    data_crs = ccrs.LambertConformal(central_longitude=-42.0) #, standard_parallels=[66.3])
    x = np.zeros(2)
    y = np.zeros(2)
    x[0], y[0] = data_crs.transform_point(lon[0], lat[0], src_crs=ccrs.PlateCarree())
    x[1], y[1] = data_crs.transform_point(lon[1], lat[1], src_crs=ccrs.PlateCarree())

    print("x0: {}, x1: {}, y1: {}, y2: {}".format(x[0], x[1], y[0], y[1]))

    nora3_da = nora3.air_pressure_at_sea_level.sel(time="2019-01-01 04:00:00",
        height_above_msl=0.0,
        x=slice(x[0], x[1]), y=slice(y[0], y[1]))
        
    print(nora3_da)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines() 
    nora3_da.plot.pcolormesh("longitude", "latitude")
    plt.show()


# %%
