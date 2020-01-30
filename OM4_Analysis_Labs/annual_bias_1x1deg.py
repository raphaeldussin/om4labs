#!/usr/bin/env python

import io
from OM4_Analysis_Labs import m6plot
from OM4_Analysis_Labs import helpers
import numpy as np
import argparse
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

possible_lon_names = ['lon', 'LON', 'longitude', 'LONGITUDE']
possible_lat_names = ['lat', 'LAT', 'latitude', 'LATITUDE']
possible_time_names = ['time', 'TIME', 'latitude']
possible_depth_names = ['z_l', 'depth', 'DEPTH']

imgbufs = []


def read_all_data(args):
    """ read data from model and obs files, process data and return it """

    # define list of possible netcdf names for given field
    if args.field == 'SST':
        possible_variable_names = ['thetao', 'temp', 'ptemp', 'TEMP', 'PTEMP']
    else:
        raise ValueError(f'field {args.field} is not available')

    # read in model and obs data
    xmodel, ymodel, datamodel = read_data(args.infile,
                                          possible_variable_names,
                                          depth=args.depth)

    xobs, yobs, dataobs = read_data(args.obs,
                                    possible_variable_names,
                                    depth=args.depth)
    # check consistency of coordinates
    assert np.allclose(xmodel, xobs)
    assert np.allclose(ymodel, yobs)

    # restrict model to where obs exists
    datamodel = datamodel.where(dataobs)

    # dump values
    model = datamodel.to_masked_array()
    obs = dataobs.to_masked_array()

    # compute area
    area = compute_area_regular_grid(args.infile)

    return xmodel, ymodel, area, model, obs


def read_data(ncfile, possible_variable_names, depth=None):
    """ read data from one file """
    ds = xr.open_dataset(ncfile)
    # find the appropriate variable names
    varname = helpers.try_variable_from_list(list(ds.variables),
                                             possible_variable_names)
    varlon = helpers.try_variable_from_list(list(ds.variables),
                                            possible_lon_names)
    varlat = helpers.try_variable_from_list(list(ds.variables),
                                            possible_lat_names)
    vartime = helpers.try_variable_from_list(list(ds.variables),
                                             possible_time_names)
    if depth is not None:
        vardepth = helpers.try_variable_from_list(list(ds.variables),
                                                  possible_depth_names)

    if varname is None:
        raise ValueError(f'no suitable variable found in {ncfile}')
    if varlon is None:
        raise ValueError(f'no suitable longitude found in {ncfile}')
    if varlat is None:
        raise ValueError(f'no suitable latitude found in {ncfile}')

    da = ds[varname]

    x = ds[varlon].values
    y = ds[varlat].values

    # extract the desired level
    if (depth is not None) and (vardepth in da.dims):
        da2d = da.sel({vardepth: depth})
    else:
        da2d = da

    # average in time if needed
    if (vartime is not None) and (vartime in da.dims):
        data = da2d.mean(dim='time')
    else:
        data = da2d

    return x, y, data


def compute_area_regular_grid(ncfile, Rearth=6378e+3):
    """ compute the cells area on a regular grid """

    ds = xr.open_dataset(ncfile)

    rfac = 2*np.pi*Rearth/360
    dx1d = rfac * (ds['lon_bnds'].isel(bnds=1) - ds['lon_bnds'].isel(bnds=0))
    dy1d = rfac * (ds['lat_bnds'].isel(bnds=1) - ds['lat_bnds'].isel(bnds=0))

    dx2d, dy2d = np.meshgrid(dx1d, dy1d)
    lon2d, lat2d = np.meshgrid(ds['lon'], ds['lat'])

    dx = dx2d * np.cos(2*np.pi*lat2d/360)
    dy = dy2d
    area = dx * dy
    return area


def get_run_name(ncfile):
    ds = xr.open_dataset(ncfile)
    if 'title' in ds.attrs:
        return ds.attrs['title']
    else:
        return 'Unknown experiment'


def plot_diff(x, y, model, obs, diff_kwargs, stream=False):
    """ make difference plot """
    if stream:
        img = io.BytesIO()
        diff_kwargs['img'] = img

    m6plot.xyplot(model - obs, x, y, **diff_kwargs)

    if stream:
        imgbufs.append(img)


def plot_compare(x, y, model, obs, compare_kwargs, stream=False):
    """ make 3 panel compare plot """
    if stream:
        img = io.BytesIO()
        compare_kwargs['img'] = img

    m6plot.xycompare(model, obs, x, y, **compare_kwargs)

    if stream:
        imgbufs.append(img)


def main():
    parser = argparse.ArgumentParser(description='Script for plotting \
                                                  annual-average bias to obs')
    parser.add_argument('-i', '--infile', type=str, required=True,
                        help='Annually-averaged file containing model data')
    parser.add_argument('-f', '--field', type=str,
                        required=True, help='field name compared to obs')
    parser.add_argument('-d', '--depth', type=float, default=None,
                        required=False, help='depth of field compared to obs')
    parser.add_argument('-l', '--label', type=str, default='',
                        required=False, help='Label to add to the plot')
    parser.add_argument('-s', '--suptitle', type=str, default='',
                        required=False, help='Super-title for experiment. \
                              Default is to read from netCDF file')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        required=False, help='output directory for plots')
    parser.add_argument('-O', '--obs', type=str, required=True,
                        help='File containing obs data to compare against')
    parser.add_argument('-D', '--dataset', type=str, required=True,
                        help='Name of the observational dataset')
    parser.add_argument('-S', '--stream', type=str, required=False,
                        help='stream output plot (diff/compare)')
    cmdLineArgs = parser.parse_args()

    streamdiff = True if cmdLineArgs.stream == 'diff' else False
    streamcompare = True if cmdLineArgs.stream == 'compare' else False

    # read the data needed for plots
    x, y, area, model, obs = read_all_data(cmdLineArgs)

    # set plots properties according to variable
    if cmdLineArgs.field == 'SST':
        var = 'SST'
        units = '[$\degree$C]'
        clim_diff = m6plot.pmCI(0.25, 4.5, .5)
        clim_compare = m6plot.linCI(-2, 29, .5)
        cmap_diff = 'dunnePM'
        cmap_compare = 'dunneRainbow'
    # elif cmdLineArgs.field == 'NEW_VAR':
    else:
        raise ValueError(f'field {cmdLineArgs.field} is not available')

    # common plot properties
    pngout = cmdLineArgs.outdir
    obsds = cmdLineArgs.dataset

    if cmdLineArgs.suptitle != '':
        suptitle = f'{cmdLineArgs.suptitle} {cmdLineArgs.label}'
    else:
        title = get_run_name(cmdLineArgs.infile)
        suptitle = f'{title} {cmdLineArgs.label}'

    title_diff = f'{var} bias (w.r.t. {obsds}) {units}'
    title1_compare = f'{var} {units}'
    title2_compare = f'{obsds} {var} {units}'
    png_diff = f'{pngout}/{var}_annual_bias_{obsds}.png'
    png_compare = f'{pngout}/{var}_annual_bias_{obsds}.3_panel.png'

    diff_kwargs = {'area': area, 'suptitle': suptitle,
                   'title': title_diff, 'clim': clim_diff,
                   'colormap': cmap_diff, 'centerlabels': True,
                   'extend': 'both', 'save': png_diff}
    compare_kwargs = {'area': area, 'suptitle': suptitle,
                      'title1': title1_compare, 'title2': title2_compare,
                      'clim': clim_compare, 'colormap': cmap_compare,
                      'extend': 'max', 'dlim': clim_diff,
                      'dcolormap': cmap_diff, 'dextend': 'both',
                      'centerdlabels': True, 'save': png_compare}

    # make diff plot
    plot_diff(x, y, model, obs, diff_kwargs, stream=streamdiff)

    # make compare plot
    plot_compare(x, y, model, obs, compare_kwargs, stream=streamcompare)

    if cmdLineArgs.stream is not None:
        return imgbufs


if __name__ == '__main__':
    main()
