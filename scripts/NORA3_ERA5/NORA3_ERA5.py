"""NORA3 and ERA5 tools. 

Install requirements with pip3 install -r requirements.txt
"""

#%%
import netCDF4 as nc4
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

def get_timeseries(param, lat, lon, start_time, end_time, use_atm=True):
    """Time series extraction from NORA3 and ERA5.

    NOTE: Use ERA5 parameter names. See atm_params_nora3 for NORA3 equivalents

    Returns tuple with string "NORA3" or "ERA5" depending on which data set was
    used, and time series xarray for parameter param at location (lat, lon) in 
    interval [start_time, end_time] with a temporal resolution of one hour. If 
    NORA3 data is not available for the given location and the full time 
    interval, ERA5 data will be used. The nearest grid point will be used.

    Data directories: 
    /lustre/storeB/project/fou/om/ERA/ERA5 [1979-1, 2019-12]
    /lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf [1997-08, 2019-12]
    """
    available_atm_params = ["msl", "u10", "v10"]
    available_wave_params = ["msl", "mwd", "mp2", "pp1d", "swh"]

    atm_params_nora3 = {
        "msl": "air_pressure_at_sea_level",
        "u10": "x_wind_10m",
        "v10": "y_wind_10m"
    }

    # sanity check arguments
    if param not in available_atm_params and param not in available_wave_params:
        raise RuntimeError("Undefined parameter: " + param)
    if datetime(2019, 12, 31) < start_time or start_time < datetime(1979, 1, 1):
        raise RuntimeError("Start time outside data set time interval")
    if datetime(2019, 12, 31) < end_time or end_time < datetime(1979, 1, 1):
        raise RuntimeError("End time outside data set time interval")

    # inside time interval and domain for NORA3?
    if start_time >= datetime(1997, 8, 1, 4, 0, 0) and (44.0 <= lat <= 83.0) and (-30.0 <= lon <= 85.0):
        return ("NORA3", get_nora3_timeseries(atm_params_nora3[param], lat, lon, start_time, end_time))
    else:
        return ("ERA5", get_era5_timeseries(param, lat, lon, start_time, end_time, use_atm))

def get_era5_timeseries(param, lat, lon, start_time, end_time, use_atm=True):
    """Time series extraction from ERA5.

    Returns time series xarray for parameter param at location (lat, lon) in interval
    [start_time, end_time] with a temporal resolution of one hour. The nearest 
    grid point will be used.

    TODO: Selection of the nearest wet (not missing_value) point for wave params using
    xarray. This is not straightforward to implement at this point
    (see also https://github.com/pydata/xarray/issues/644)

    Data directories: 
    /lustre/storeB/project/fou/om/ERA/ERA5 [1979-1, 2019-12]
    """
    data_dir = "/lustre/storeB/project/fou/om/ERA/ERA5"
    available_atm_params = ["msl", "u10", "v10"]
    available_wave_params = ["msl", "mwd", "mp2", "pp1d", "swh"]
    
    # sanity check arguments
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
            filenames.append(os.path.join(data_dir, "atm/era5_atm_CDS_{}.nc".format(month.strftime("%Y%m"))))
    elif param in available_wave_params:
        for month in (start_time + relativedelta(months=+n) for n in range(month_count)):
            filenames.append(os.path.join(data_dir, "wave/era5_wave_CDS_{}.nc".format(month.strftime("%Y%m"))))
    else:
        raise RuntimeError(param + " is not found in ERA5 wave data set (try use_atm=True)")
    
    era5 = xr.open_mfdataset(filenames, parallel=True)

    # extract data set
    era5_da = era5[param].sel(longitude=lon, latitude=lat, method="nearest")
    era5_da = era5_da.sel(time=slice(start_time, end_time))

    # return time series as xarray
    return era5_da

def get_nora3_timeseries(param, lat, lon, start_time, end_time):
    """Time series extraction from NORA3.

    Returns time series xarray for parameter param at location (lat, lon) in interval
    [start_time, end_time] with a temporal resolution of one hour. The nearest 
    grid point will be used.

    Data directories: 
    /lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf [1997-08, 2019-12]
    """
    data_dir = "/lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf"
    available_atm_params = ["air_pressure_at_sea_level", "x_wind_10m", "y_wind_10m"]

    # sanity check arguments
    if param not in available_atm_params:
        raise RuntimeError("Undefined parameter: " + param)
    if 44.0 > lat > 83.0:
        raise RuntimeError("Latitude (lat) must be in the interval [44.0, 83.0]")
    if -30.0 > lon > 85.0:
        raise RuntimeError("Longitude (lon) must be in the interval [-30.0, 85.0]")
    
    #print("From " + start_time.strftime("%Y%m%d-%H"))
    #print("To " + end_time.strftime("%Y%m%d-%H"))

    # find and open correct netCDF file(s)
    filenames = []
    hour = timedelta(hours=1)
    current_time = start_time

    if param in available_atm_params:
        while current_time <= end_time:
            current_time_with_offset = current_time - timedelta(hours=4)
            
            # find correct period folder
            if current_time_with_offset.hour < 6:
                period = 0
            elif current_time_with_offset.hour < 12:
                period = 6
            elif current_time_with_offset.hour < 18:
                period = 12
            else:
                period = 18
            
            # find correct index file
            index_file = current_time.hour - period
            if index_file < 0:
                index_file += 24

            filenames.append(
                os.path.join(data_dir, "{year}/{month}/{day}/{period:02d}/fc{year}{month}{day}{period:02d}_00{index_file}_fp.nc" \
                .format(year=current_time_with_offset.strftime("%Y"), 
                        month=current_time_with_offset.strftime("%m"), 
                        day=current_time_with_offset.strftime("%d"), 
                        period=period, index_file=index_file)))
            current_time += hour
    else:
        raise RuntimeError(param + " is not found in NORA3 data set")
    
    nora3 = xr.open_mfdataset(filenames)

    # find coordinates in data set projection by transformation:
    #data_crs = ccrs.LambertConformal(central_longitude=-42.0, central_latitude=66.3,
    #            standard_parallels=[66.3, 66.3], 
    #            globe=ccrs.Globe(datum="WGS84",
    #            semimajor_axis=6371000.0))
    #x, y = data_crs.transform_point(lon, lat, src_crs=ccrs.PlateCarree())

    #nora3_da_lon = nora3["longitude"].sel(x=x, y=y, method="nearest")
    #nora3_da_lat = nora3["latitude"].sel(x=x, y=y, method="nearest")
    #print("Projected lon, lat: " + str(nora3_da_lon.values) + ", " + str(nora3_da_lat.values))
    
    # find coordinates in data set projection by lookup in lon-lat variables
    abslat = np.abs(nora3.latitude-lat)
    abslon = np.abs(nora3.longitude-lon)
    cor = np.maximum(abslon, abslat)
    ([y_idx], [x_idx]) = np.where(cor == np.min(cor))

    #print("Projected lon, lat: " 
    #        + str(nora3["longitude"].isel(x=x_idx, y=y_idx).values) + ", " 
    #        + str(nora3["latitude"].isel(x=x_idx, y=y_idx).values))

    # extract data set
    #nora3_da = nora3[param].sel(x=x, y=y, method="nearest")
    nora3_da = nora3[param].isel(x=x_idx, y=y_idx)
    nora3_da = nora3_da.sel(time=slice(start_time, end_time))

    # return time series as xarray
    return nora3_da

# test time series extraction and plotting from NORA3/ERA5 data set
ds, da = get_timeseries("u10", 58.0, 20.0, 
        datetime(1997, 8, 1, 4, 0, 0), datetime(1997, 8, 1, 5, 0, 0))
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
