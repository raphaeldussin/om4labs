#!/usr/bin/env python

import io
import numpy as np
import argparse
import xarray as xr
import warnings
try:
    from OM4_Analysis_Labs import m6plot
    from OM4_Analysis_Labs.helpers import get_run_name, try_variable_from_list
    from OM4_Analysis_Labs.om4plotting import plot_xydiff, plot_xycompare
    from OM4_Analysis_Labs.om4compute import read_data, subset_data
    from OM4_Analysis_Labs.om4compute import simple_average, copy_coordinates
    from OM4_Analysis_Labs.om4compute import compute_area_regular_grid
except ImportError:
    # DORA mode, works without install.
    # reads from current directory
    import m6plot
    from helpers import get_run_name, try_variable_from_list
    from om4plotting import plot_xydiff, plot_xycompare
    from om4compute import read_data, subset_data
    from om4compute import simple_average, copy_coordinates
    from om4compute import compute_area_regular_grid

warnings.filterwarnings("ignore")

surface_default_depth = 2.5  # meters, first level of 1x1deg grid

imgbufs = []


def read_all_data(args, **kwargs):
    """ read data from model and obs files, process data and return it """

    # define list of possible netcdf names for given field
    if args.field == 'SST':
        possible_variable_names = ['thetao', 'temp', 'ptemp', 'TEMP', 'PTEMP']
    else:
        raise ValueError(f'field {args.field} is not available')

    dsmodel = xr.open_mfdataset(args.infile, combine='by_coords', decode_times=False)
    dsobs = xr.open_mfdataset(args.obs, combine='by_coords', decode_times=False)

    # read in model and obs data
    datamodel = read_data(dsmodel, possible_variable_names)
    dataobs = read_data(dsobs, possible_variable_names)

    # subset data
    if (args.depth is not None):
        datamodel = subset_data(datamodel, 'assigned_depth', args.depth)
        dataobs = subset_data(dataobs, 'assigned_depth', args.depth)

    # reduce data along depth (not yet implemented)
    if 'depth_reduce' in kwargs:
        if kwargs['depth_reduce'] == 'mean':
            # do mean
            pass
        elif kwargs['depth_reduce'] == 'sum':
            # do mean
            pass

    # reduce data along time, here mandatory
    datamodel = simple_average(datamodel, 'assigned_time')
    dataobs = simple_average(dataobs, 'assigned_time')

    datamodel = datamodel.squeeze()
    dataobs = dataobs.squeeze()

    # check final data is 2d
    assert len(datamodel.dims) == 2
    assert len(dataobs.dims) == 2

    # check consistency of coordinates
    assert np.allclose(datamodel['assigned_lon'], dataobs['assigned_lon'])
    assert np.allclose(datamodel['assigned_lat'], dataobs['assigned_lat'])

    # homogeneize coords
    dataobs = copy_coordinates(datamodel, dataobs,
                               ['assigned_lon', 'assigned_lat'])

    # restrict model to where obs exists
    datamodel = datamodel.where(dataobs)

    # dump values
    model = datamodel.to_masked_array()
    obs = dataobs.to_masked_array()
    x = datamodel['assigned_lon'].values
    y = datamodel['assigned_lat'].values

    # compute area
    if 'areacello' in dsmodel.variables:
        area = ds['areacello'].values
    else:
        if model.shape == (180, 360):
            area = compute_area_regular_grid(dsmodel)
        else:
            raise IOError('no cell area provided')

    return x, y, area, model, obs


def run():
    """ parse the command line arguments """
    parser = argparse.ArgumentParser(description='Script for plotting \
                                                  annual-average bias to obs')
    parser.add_argument('-i', '--infile', type=str, nargs='+', required=True,
                        help='Annually-averaged file(s) containing model data')
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
    # execute the main code
    main(cmdLineArgs)


def main(cmdLineArgs):
    """ main can be called from either command line and then use parser from run()
    or DORA can build the args and run it directly """

    streamdiff = True if cmdLineArgs.stream == 'diff' else False
    streamcompare = True if cmdLineArgs.stream == 'compare' else False
    streamnone = True if cmdLineArgs.stream == None else False

    # set plots properties according to variable
    if cmdLineArgs.field == 'SST':
        var = 'SST'
        units = '[$\degree$C]'
        clim_diff = m6plot.pmCI(0.25, 4.5, .5)
        clim_compare = m6plot.linCI(-2, 29, .5)
        cmap_diff = 'dunnePM'
        cmap_compare = 'dunneRainbow'
        if cmdLineArgs.depth is None:
            cmdLineArgs.depth = surface_default_depth
    # elif cmdLineArgs.field == 'NEW_VAR':
    else:
        raise ValueError(f'field {cmdLineArgs.field} is not available')

    # read the data needed for plots
    x, y, area, model, obs = read_all_data(cmdLineArgs)

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
    if streamdiff or streamnone:
        img = plot_xydiff(x, y, model, obs, diff_kwargs, stream=streamdiff)
        imgbufs = [img]

    # make compare plot
    if streamcompare or streamnone:
        img = plot_xycompare(x, y, model, obs, compare_kwargs, stream=streamcompare)
        imgbufs = [img]

    if cmdLineArgs.stream is not None:
        return imgbufs


if __name__ == '__main__':
    run()
