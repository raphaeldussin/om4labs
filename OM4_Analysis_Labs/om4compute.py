# need a data_read, data_sel, data_reduce

import numpy as np

try:
    from OM4_Analysis_Labs.helpers import try_variable_from_list
except ImportError:
    # DORA mode, works without install.
    # reads from current directory
    from helpers import try_variable_from_list


possible_names = {}
possible_names['lon'] = ['lon', 'LON', 'longitude', 'LONGITUDE']
possible_names['lat'] = ['lat', 'LAT', 'latitude', 'LATITUDE']
possible_names['time'] = ['time', 'TIME', 'latitude']
possible_names['depth'] = ['z_l', 'depth', 'DEPTH']
possible_names['interfaces'] = ['z_i']


def infer_and_assign_coord(ds, da, coordname):
    """ infer what the coord name is and assign it to dataarray """
    assigned_coordname = try_variable_from_list(list(ds.variables),
                                                possible_names[coordname])
    if (assigned_coordname is not None) and (assigned_coordname in da.dims):
        da = da.rename({assigned_coordname: f'assigned_{coordname}'})
    return da


def read_data(ds, possible_variable_names):
    """ read data from one file """

    # find the appropriate variable names
    varname = try_variable_from_list(list(ds.variables),
                                     possible_variable_names)
    if varname is None:
        raise ValueError(f'no suitable variable found in {ncfile}')

    da = ds[varname]
    da = infer_and_assign_coord(ds, da, 'lon')
    da = infer_and_assign_coord(ds, da, 'lat')
    da = infer_and_assign_coord(ds, da, 'time')
    da = infer_and_assign_coord(ds, da, 'depth')
    da = infer_and_assign_coord(ds, da, 'interfaces')

    return da


def subset_data(da, coordname, subset):
    """ subset (float or slice) dataarray along coord """
    if coordname in da.coords:
        da = da.sel({coordname: subset})
    return da


def simple_average(da, coordname):
    """ average """
    if coordname in da.coords:
        da = da.mean(dim=coordname)
    return da


def copy_coordinates(da1, da2, coords):
    """ copy coordinates of da1 into da2 """

    for coord in coords:
        da2[coord] = da1[coord]

    return da2


def compute_area_regular_grid(ds, Rearth=6378e+3):
    """ compute the cells area on a regular grid """

    rfac = 2*np.pi*Rearth/360

    up = {'bnds': 1}
    down = {'bnds': 0}
    if 'time' in ds['lon_bnds'].dims:
        up.update({'time': 0})
        down.update({'time': 0})

    dx1d = rfac * (ds['lon_bnds'].isel(up) - ds['lon_bnds'].isel(down))
    dy1d = rfac * (ds['lat_bnds'].isel(up) - ds['lat_bnds'].isel(down))

    dx2d, dy2d = np.meshgrid(dx1d, dy1d)
    lon2d, lat2d = np.meshgrid(ds['lon'].values, ds['lat'].values)

    dx = dx2d * np.cos(2*np.pi*lat2d/360)
    dy = dy2d
    area = dx * dy
    return area
