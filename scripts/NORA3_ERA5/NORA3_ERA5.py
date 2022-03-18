"""NORA3 and ERA5 tools. 

Install requirements with pip3 install -r requirements.txt

NOTE: expver is not treated, and files containing the expver dimension cannot be used
(see https://confluence.ecmwf.int/pages/viewpage.action?pageId=173385064 for details about expver)
"""

#%%
import netCDF4 as nc4
import numpy as np
import os
import argparse
import xarray as xr
import cartopy.crs as ccrs
import dask
import gc
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

def _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size):
    """For time deltas, we must ensure that we use the same encoding as
    what was previously stored.
    We likely need to do this as well for variables that had custom
    econdings too

    Author: Mark Harfouche (https://github.com/pydata/xarray/issues/1672#issuecomment-685222909)
    """
    if hasattr(nc_variable, 'calendar'):
        
        data.encoding = {
            'units': nc_variable.units,
            'calendar': nc_variable.calendar,
        }

    data_encoded = xr.conventions.encode_cf_variable(data) # , name=name)
    
    left_slices = data.dims.index(expanding_dim)
    right_slices = data.ndim - left_slices - 1
    nc_slice   = (slice(None),) * left_slices + (slice(nc_shape, nc_shape + added_size),) + (slice(None),) * (right_slices)
    nc_variable[nc_slice] = data_encoded.data
        
def append_to_netcdf(filename, ds_to_append, unlimited_dims):
    """Append an xarray DataSet to unlimited_dim(s) in an existing netCDF file.
    
    Author: Mark Harfouche (https://github.com/pydata/xarray/issues/1672#issuecomment-685222909)
    """
    if isinstance(unlimited_dims, str):
        unlimited_dims = [unlimited_dims]
        
    if len(unlimited_dims) != 1:
        # TODO: change this so it can support multiple expanding dims
        raise ValueError(
            "We only support one unlimited dim for now, "
            "got {}.".format(len(unlimited_dims)))

    unlimited_dims = list(set(unlimited_dims))
    expanding_dim = unlimited_dims[0]
    
    with nc4.Dataset(filename, mode='a') as nc:
        nc_dims = set(nc.dimensions.keys())

        nc_coord = nc[expanding_dim]
        nc_shape = len(nc_coord)
        
        added_size = len(ds_to_append[expanding_dim])
        variables, attrs = xr.conventions.encode_dataset_coordinates(ds_to_append)

        for name, data in variables.items():
            if expanding_dim not in data.dims:
                # Nothing to do, data assumed to the identical
                continue

            nc_variable = nc[name]
            _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size)

def get_closest_water_point(start_i, start_j, data, missing_value=-32767):
    """
    Code from josteinb@met.no (adapted)

    desc:
        Breadth first search function to find index of nearest
        non-missing_value point
    args:
        - start_i: Start index of i
        - start_j: Start index of j
        - data: grid with data
        - missing_value: value of missing_value for paramter
    return:
        - index of point
    """
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = set()
    q = [(start_i, start_j)]    # init queue to start pos
    count = 0

    # while something in queue
    while q:
        current = q.pop(0)      # pop the first in waiting queue
        # if we have visited this before
        if current in visited:
            continue
        visited.add(current)    # Add to set of visited
        # If not in border list
        # Test if this is land, if true go to next in queue, else return idx
        if data[current[0], current[1]] != missing_value:
            return current[0], current[1]
        count += 1      # updates the count
        # Loop over neighbours and add to queue
        for di, dj in dirs:
            new_i = current[0]+di
            new_j = current[1]+dj
            q.append((new_i, new_j))

def get_timeseries(param, lon, lat, start_time, end_time, use_atm=True):
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

def get_era5_timeseries(param, lon, lat, start_time, end_time, use_atm=True):
    """Time series extraction from ERA5.

    Returns time series xarray for parameter param at location (lat, lon) in interval
    [start_time, end_time] with a temporal resolution of one hour. The nearest 
    grid point will be used.

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

def get_nora3_timeseries(param, lon, lat, start_time, end_time):
    """Time series extraction from NORA3.

    Returns time series xarray for parameter param at location (lat, lon) in interval
    [start_time, end_time] with a temporal resolution of one hour. The nearest 
    grid point will be used.

    Data directories: 
    /lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf [1997-08, 2019-12]
    """
    data_dir = "/lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf"
    available_atm_params = ["air_pressure_at_sea_level", "x_wind_10m", "y_wind_10m", 
        "integral_of_toa_net_downward_shortwave_flux_wrt_time", 
        "integral_of_surface_net_downward_shortwave_flux_wrt_time",
        "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
        "snowfall_amount_acc", 
        "precipitation_amount_acc",
        "air_temperature_2m",
        "relative_humidity_2m",
        "cloud_area_fraction",
        "convective_cloud_area_fraction",
        "high_type_cloud_area_fraction",
        "medium_type_cloud_area_fraction",
        "low_type_cloud_area_fraction"]
    integrated_params = ["integral_of_toa_net_downward_shortwave_flux_wrt_time", 
        "integral_of_surface_net_downward_shortwave_flux_wrt_time",
        "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
        "snowfall_amount_acc", 
        "precipitation_amount_acc"]

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
    first_timestep_filename = ""
    filenames = []
    spinup_filenames = []
    hour = timedelta(hours=1)
    current_time = start_time

    # get the time step before start_time needed for compute the intantanous value for start_time
    if param in integrated_params:
        current_time_with_offset = current_time - hour - timedelta(hours=4)
            
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
        index_file = current_time.hour - 1 - period
        if index_file < 0:
            index_file += 24
            
        first_timestep_filename = os.path.join(data_dir, "{year}/{month}/{day}/{period:02d}/fc{year}{month}{day}{period:02d}_00{index_file}_fp.nc" \
            .format(year=current_time_with_offset.strftime("%Y"), 
                    month=current_time_with_offset.strftime("%m"), 
                    day=current_time_with_offset.strftime("%d"), 
                    period=period, index_file=index_file))

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
            
            # build list of spinup filenames – the spinup timestep will only be used to get the 
            # intantanous value for the next timestep
            if param in integrated_params and index_file == 4:
                spinup_filenames.append(
                    os.path.join(data_dir, "{year}/{month}/{day}/{period:02d}/fc{year}{month}{day}{period:02d}_00{index_file}_fp.nc" \
                    .format(year=current_time_with_offset.strftime("%Y"), 
                            month=current_time_with_offset.strftime("%m"), 
                            day=current_time_with_offset.strftime("%d"), 
                            period=period, index_file=3)))

            current_time += hour
    else:
        raise RuntimeError(param + " is not found in NORA3 data set")
    
    #print(first_timestep_filename)
    #print(spinup_filenames)
    #print(filenames)

    # correction for not being able to read files with open_mfdataset
    # some leap year oddity??
    filenames = [x for x in filenames if not x.startswith("/lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf/2020/02/29/18/fc2020022918")]
    spinup_filenames = [x for x in spinup_filenames if not x.startswith("/lustre/storeB/project/fou/om/WINDSURFER/HM40h12/netcdf/2020/02/29/18/fc2020022918")]

    #print(spinup_filenames)
    #print(filenames)

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
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

        del nora3_da["x"]
        del nora3_da["y"]
        del nora3_da.attrs["grid_mapping"]

        if param in integrated_params:
            nora3_first_timestep = xr.open_dataset(first_timestep_filename)
            nora3_da_first_timestep = nora3_first_timestep[param].isel(x=x_idx, y=y_idx)
            nora3_da_first_timestep = nora3_da_first_timestep.sel(time=slice(start_time-hour))

            nora3_spinup = xr.open_mfdataset(spinup_filenames)
            nora3_da_spinup = nora3_spinup[param].isel(x=x_idx, y=y_idx)
            nora3_da_spinup = nora3_da_spinup.sel(time=slice(start_time, end_time-hour))

            # load data - this will not work particularly well for long time series...
            # XXX: should figure out a better solution
            nora3_da.load()
            nora3_da_spinup.load()

            # add zero-values for missing times
            # XXX: extremely hacky workaround
            if end_time == datetime(2020, 2, 29, 23):
                nora3_da_new = xr.DataArray(data=np.zeros((2,1)), dims=["time", "height0"],
                        coords={"time": [datetime(2020, 2, 29, 22), datetime(2020, 2, 29, 23)],
                                "height0": [0.0]})
                nora3_da = xr.concat([nora3_da, nora3_da_new], dim="time")
                
                nora3_da_spinup_new = xr.DataArray(data=np.zeros((1,1)), dims=["time", "height0"],
                        coords={"time": [datetime(2020, 2, 29, 21)],
                                "height0": [0.0]})
                nora3_da_spinup = xr.concat([nora3_da_spinup, nora3_da_spinup_new], dim="time")
            ###
            if start_time == datetime(2020, 3, 1, 0) :
                nora3_da_new = xr.DataArray(data=np.zeros((4,1)), dims=["time", "height0"],
                        coords={"time": [datetime(2020, 3, 1, 0), datetime(2020, 3, 1, 1), 
                        datetime(2020, 3, 1, 2), datetime(2020, 3, 1, 3)],
                                "height0": [0.0]})
                nora3_da = xr.concat([nora3_da_new, nora3_da], dim="time", combine_attrs="no_conflicts")
            ###

            nora3_da_original = nora3_da.copy(deep=True) 

            # load data - this will not work particularly well for long time series...
            # XXX: should figure out a better solution
            nora3_da_original.load()

            # compute instantanous values for all first timesteps after spinup (04, 10, 16, and 22)
            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 4])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 4).values \
                                                                    -nora3_da_spinup.sel(time=nora3_da_spinup.time.dt.hour == 3).values
            
            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 10])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 10).values \
                                                                    -nora3_da_spinup.sel(time=nora3_da_spinup.time.dt.hour == 9).values
            
            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 16])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 16).values \
                                                                    -nora3_da_spinup.sel(time=nora3_da_spinup.time.dt.hour == 15).values
                                                                    
            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 22])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 22).values \
                                                                    -nora3_da_spinup.sel(time=nora3_da_spinup.time.dt.hour == 21).values

            # compute instantanous values for all remaining timesteps
            midnight_indices = nora3_da.time[nora3_da.time.dt.hour == 0]
            nora3_da.loc[dict(time=midnight_indices[1:])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 0).values[1:] \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 23).values[:-1]

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 1])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 1).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 0).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 2])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 2).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 1).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 3])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 3).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 2).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 5])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 5).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 4).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 6])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 6).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 5).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 7])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 7).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 6).values
                                                                    
            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 8])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 8).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 7).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 9])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 9).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 8).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 11])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 11).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 10).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 12])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 12).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 11).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 13])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 13).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 12).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 14])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 14).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 13).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 15])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 15).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 14).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 17])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 17).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 16).values
                                                                    
            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 18])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 18).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 17).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 19])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 19).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 18).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 20])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 20).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 19).values

            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 21])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 21).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 20).values
                                                                    
            nora3_da.loc[dict(time=nora3_da.time[nora3_da.time.dt.hour == 23])] = nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 23).values \
                                                                    -nora3_da_original.sel(time=nora3_da_original.time.dt.hour == 22).values

            # compute first instantanous values
            nora3_da.values[0] = nora3_da_original.values[0] - nora3_da_first_timestep.values[0]

            # leap year problem with open_mfdataset
            # XXX: extremely hacky workaround
            if(nora3_da.values[0] < 0):
                nora3_da.values[0] = 0.0

    return nora3_da

import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                usage[2]/1024.0 )

def write_SUNPOINT_timeseries(stations, output_file, param, start_time, end_time):
    station_ids = stations["stationid"]
    station_lons = stations["longitude"]
    station_lats = stations["latitude"]

    print("From " + start_time.strftime("%Y%m%d-%H%M"))
    print("To " + end_time.strftime("%Y%m%d-%H%M"))

    stride_start_time = start_time - timedelta(hours=1)
    stride_end_time = start_time - timedelta(hours=1)
    # stride in time
    while stride_end_time is not end_time:
        print(using())
        stride_start_time = stride_end_time + timedelta(hours=1)
        stride_end_time = stride_start_time + timedelta(days=10) - timedelta(hours=1)
        if stride_end_time > end_time:
            stride_end_time = end_time

        print("From " + stride_start_time.strftime("%Y%m%d-%H%M"))
        print("To " + stride_end_time.strftime("%Y%m%d-%H%M"))

        dataarrays = []

        for (station_id, station_lon, station_lat) in zip(station_ids, station_lons, station_lats):
            print("Writing timeseries for station " + str(station_id) + " at " 
                    + str(station_lon) + ", " + str(station_lat))

            da = get_nora3_timeseries(param, station_lon, station_lat, 
                    stride_start_time, stride_end_time)
            
            dataarrays.append(da)

        combined = xr.concat(dataarrays, dim="station")
        out_da = xr.Dataset()
        out_da[param] = combined
        
        out_da = out_da.chunk(chunks={"station": 1})

        if not os.path.isfile(output_file):
            init_netcdf_output_file(out_da, station_ids, station_lons, station_lats)

            # We are not getting these variables from an existing nc-file, and therefore need to 
            # ensure that the correct dimension (station) is used
            out_da["stationid"] = out_da["stationid"].swap_dims({"stationid": "station"})
            #out_da["longitude"] = out_da["longitude"].expand_dims(dim="station")
            out_da["longitude_station"] = out_da["longitude_station"].swap_dims({"longitude_station": "station"})
            #out_da["latitude"] = out_da["latitude"].expand_dims(dim="station")
            out_da["latitude_station"] = out_da["latitude_station"].swap_dims({"latitude_station": "station"})

            out_da.to_netcdf(output_file, 
                                format="NETCDF4", engine="netcdf4", unlimited_dims="time", mode="w",
                                encoding={param: {"dtype": "float32", "zlib": False, "_FillValue": 1.0e37}})
        else:
            out_da[param].encoding['dtype'] = "float32"
            out_da[param].encoding['zlib'] = False
            out_da[param].encoding['_FillValue'] = 1.0e37

            append_to_netcdf(output_file, out_da, unlimited_dims="time")

        del combined
        del out_da
        gc.collect()

def init_netcdf_output_file(out_da, station_ids, station_lons, station_lats):
    """Initiate netCDF file with observation stations and time as unlimited dimension."""

    if(isinstance(station_ids[0], str)):
        out_da["stationid"] = station_ids
    else:
        out_da["stationid"] = station_ids.astype(str)

    out_da["longitude_station"] = station_lons
    out_da["latitude_station"] = station_lats

    # ensure CF compliance
    out_da.attrs["Conventions"] = "CF-1.8"
    out_da["longitude"].attrs["units"] = "degrees_east"
    out_da["longitude"].attrs["long_name"] = "longitude"
    out_da["longitude"].attrs["description"] = "longitude of closest data point to station"

    out_da["longitude_station"].attrs["units"] = "degrees_east"
    out_da["longitude_station"].attrs["long_name"] = "longitude_station"

    out_da["latitude"].attrs["units"] = "degrees_north"
    out_da["latitude"].attrs["long_name"] = "latitude"
    out_da["latitude"].attrs["description"] = "latitude of closest data point to station"

    out_da["latitude_station"].attrs["units"] = "degrees_north"
    out_da["latitude_station"].attrs["long_name"] = "latitude_station"

def write_timeseries(stations_file, output_file, param, start_time, end_time):
    """WiP: Get stations (w/locations), do time series extraction from ERA5/NORA3, and write results to netCDF file."""
    # write msl timeseries for the complete ERA5 period for
    # every observation in obs data file (see line below)
    stations = xr.open_mfdataset(stations_file)

    station_ids = stations["stationid"]
    station_lons = stations["longitude"]
    station_lats = stations["latitude"]

    print("From " + start_time.strftime("%Y%m%d-%H%M"))
    print("To " + end_time.strftime("%Y%m%d-%H%M"))

    stride_start_time = start_time - timedelta(hours=1)
    stride_end_time = start_time - timedelta(hours=1)
    # stride in time
    while stride_end_time is not end_time:
        stride_start_time = stride_end_time + timedelta(hours=1)
        stride_end_time = stride_start_time + timedelta(days=365) - timedelta(hours=1)
        if stride_end_time > end_time:
            stride_end_time = end_time

        print("From " + stride_start_time.strftime("%Y%m%d-%H%M"))
        print("To " + stride_end_time.strftime("%Y%m%d-%H%M"))

        dataarrays = []

        for (station_id, station_lon, station_lat) in zip(station_ids, station_lons, station_lats):
            print("Writing timeseries for station " + str(station_id.values) + " at " 
                    + str(station_lon.values) + ", " + str(station_lat.values))

            da = get_era5_timeseries(param, station_lon, station_lat, 
                    stride_start_time, stride_end_time)
            
            dataarrays.append(da)

        combined = xr.concat(dataarrays, dim="station")
        out_da = xr.Dataset()
        out_da[param] = combined
        
        out_da = out_da.chunk(chunks={"station": 1})
        print(out_da)

        if not os.path.isfile(output_file):
            init_netcdf_output_file(out_da, station_ids, station_lons, station_lats)
            out_da.to_netcdf(output_file, 
                                format="NETCDF4", engine="netcdf4", unlimited_dims="time", mode="w",
                                encoding={param: {"dtype": "float32", "zlib": False, "_FillValue": 1.0e37}})
        else:
            out_da[param].encoding['dtype'] = "float32"
            out_da[param].encoding['zlib'] = False
            out_da[param].encoding['_FillValue'] = 1.0e37

            del out_da[param].encoding['missing_value']
            del out_da[param].encoding['scale_factor']
            del out_da[param].encoding['add_offset']

            append_to_netcdf(output_file, out_da, unlimited_dims="time")

if __name__ == "__main__":
    # TODO: fix memory prob with to_netcdf() in order to write long timeseries w/o appending,
    #       find nearest "wet point" and describe difference in latlon for these stations, 
    #       extract functions (choose time stride, parameter, input and output filenames)

    # parse optional arguments
    #parser = argparse.ArgumentParser(description="Extract timeseries from NORA3/ERA5 \
    #    data sets based on location and time interval fetched from netCDF-files containing \
    #    stations, and write to netCDF.")
    #parser.add_argument('-i','--input-stations', metavar='FILENAME', \
    #    help='input file name containing stations (netCDF format)',required=False)
    #parser.add_argument('-o','--output-file', metavar='FILENAME', \
    #    help='output file',required=False)
    #parser.add_argument('-p','--parameter', metavar='PARAMETER', \
    #    help='parameter name (netCDF name)',required=False)
    #parser.add_argument('-s','--start-time', metavar='YYYY-MM-DDTHH:MM', \
    #    help='input file name containing stations (netCDF format)',required=False)
    #parser.add_argument('-e','--end-time', metavar='YYYY-MM-DDTHH:MM', \
    #    help='input file name containing stations (netCDF format)',required=False)

    #args = parser.parse_args()

    #if not len(sys.argv) > 1:
    #    print("Provide arguments. See NORA3_ERA5.py --help")

    #start_time = datetime.strptime(args.start_time, '%Y-%m-%dT%H:%M.%f')
    #end_time = datetime.strptime(args.end_time, '%Y-%m-%dT%H:%M.%f')
    #write_timeseries(args.input_stations, args.output_file, args.param, start_time, end_time)

    # hardcoded example - extracting param from start_time to end_time for all stations contained in the input_stations nedCDF file
    #input_stations = "/lustre/storeB/project/IT/geout/machine-ocean/prepared_datasets/storm_surge/aggregated_water_level_data/aggregated_water_level_observations_with_pytide_prediction_dataset.nc4"
    #output_file = "aggregated_era5_mwd.nc"
    #param = "mwd"
    #start_time = datetime(2020, 1, 1, 0)
    #end_time = datetime(2021, 4, 30, 23)
    #write_timeseries(input_stations, output_file, param, start_time, end_time)

    # hardcoded example - extracting param from start_time to end_time for all stations contained in the input_stations nedCDF file
    #input_stations = {
    #    "stationid": ["Blindern", "Bergen", "Tromsø-Holt"],
    #    "longitude": [10.72, 5.332, 18.9368],
    #    "latitude": [59.9423, 60.3837, 69.6537]
    #}
    #input_stations = {
    #    "stationid": ["Blindern", "Bergen", "Tromsø-Holt"],
    #    "longitude": [10.72, 5.332, 18.9368],
    #    "latitude": [59.9423, 60.3837, 69.6537]
    #}
    input_stations = {
        "stationid": ["Bergen", "Tromsø-Holt"],
        "longitude": [5.332, 18.9368],
        "latitude": [60.3837, 69.6537]
    }
    output_file = "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time_2020.nc"
    param = "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time"
    start_time = datetime(2020, 1, 1, 0)
    end_time = datetime(2020, 12, 31, 23)
    write_SUNPOINT_timeseries(input_stations, output_file, param, start_time, end_time)

# %%
