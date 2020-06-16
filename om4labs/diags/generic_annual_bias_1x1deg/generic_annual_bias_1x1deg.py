#!/usr/bin/env python

import numpy as np
import argparse
import xarray as xr
import warnings
import pkg_resources as pkgr
import intake

try:
    from om4labs import m6plot
    from om4labs.helpers import get_run_name, try_variable_from_list
    from om4labs.om4plotting import plot_xydiff, plot_xycompare
    from om4labs.om4common import read_data, subset_data
    from om4labs.om4common import simple_average, copy_coordinates
    from om4labs.om4common import compute_area_regular_grid
except ImportError:
    # DORA mode, works without install.
    # reads from current directory
    import m6plot
    from helpers import get_run_name, try_variable_from_list
    from om4plotting import plot_xydiff, plot_xycompare
    from om4common import read_data, subset_data
    from om4common import simple_average, copy_coordinates
    from om4common import compute_area_regular_grid

imgbufs = []


def read(dictArgs):
    """ read data from model and obs files, process data and return it """

    dsmodel = xr.open_mfdataset(dictArgs["infile"],
                                combine="by_coords",
                                decode_times=False)

    if dictArgs["obsfile"] is not None:
        # priority to user-provided obs file
        dsobs = xr.open_mfdataset(dictArgs["obsfile"],
                                  combine="by_coords",
                                  decode_times=False)
    else:
        # use dataset from catalog, either from command line or default
        cat_platform = "catalogs/obs_catalog_" + dictArgs["platform"] + ".yml"
        catfile = pkgr.resource_filename("om4labs", cat_platform)
        cat = intake.open_catalog(catfile)
        dsobs = cat[dictArgs["dataset"]].to_dask()

    # read in model and obs data
    datamodel = read_data(dsmodel, dictArgs["possible_variable_names"])
    dataobs = read_data(dsobs, dictArgs["possible_variable_names"])

    # subset data
    if dictArgs["depth"] is None:
        dictArgs["depth"] = dictArgs["surface_default_depth"]

    if dictArgs["depth"] is not None:
        datamodel = subset_data(datamodel, "assigned_depth", dictArgs["depth"])
        dataobs = subset_data(dataobs, "assigned_depth", dictArgs["depth"])

    # reduce data along depth (not yet implemented)
    if "depth_reduce" in dictArgs:
        if dictArgs["depth_reduce"] == "mean":
            # do mean
            pass
        elif dictArgs["depth_reduce"] == "sum":
            # do sum
            pass

    # reduce data along time, here mandatory
    if ("assigned_time" in datamodel.dims) and (len(datamodel["assigned_time"]) > 1):
        warnings.warn(
            "input dataset has more than one time record, "
            + "performing non-weighted average"
        )
        datamodel = simple_average(datamodel, "assigned_time")
    if ("assigned_time" in dataobs.dims) and len(dataobs["assigned_time"]) > 1:
        warnings.warn(
            "reference dataset has more than one time record, "
            + "performing non-weighted average"
        )
        dataobs = simple_average(dataobs, "assigned_time")

    datamodel = datamodel.squeeze()
    dataobs = dataobs.squeeze()

    # check final data is 2d
    assert len(datamodel.dims) == 2
    assert len(dataobs.dims) == 2

    # check consistency of coordinates
    assert np.allclose(datamodel["assigned_lon"], dataobs["assigned_lon"])
    assert np.allclose(datamodel["assigned_lat"], dataobs["assigned_lat"])

    # homogeneize coords
    dataobs = copy_coordinates(datamodel,
                               dataobs,
                               ["assigned_lon", "assigned_lat"])

    # restrict model to where obs exists
    datamodel = datamodel.where(dataobs)

    # dump values
    model = datamodel.to_masked_array()
    obs = dataobs.to_masked_array()
    x = datamodel["assigned_lon"].values
    y = datamodel["assigned_lat"].values

    # compute area
    if "areacello" in dsmodel.variables:
        area = dsmodel["areacello"].values
    else:
        if model.shape == (180, 360):
            area = compute_area_regular_grid(dsmodel)
        else:
            raise IOError("no cell area provided")

    return x, y, area, model, obs


def parse(cliargs=None):
    """ parse the command line arguments """
    parser = argparse.ArgumentParser(
        description="Script for plotting \
                                                  annual-average bias to obs"
    )
    parser.add_argument(
        "infile",
        metavar="INFILE",
        type=str,
        nargs="+",
        help="Annually-averaged file(s) containing model data",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=float,
        default=None,
        required=False,
        help="depth of field compared to obs",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="",
        required=False,
        help="Label to add to the plot",
    )
    parser.add_argument(
        "-s",
        "--suptitle",
        type=str,
        default="",
        required=False,
        help="Super-title for experiment. \
                              Default is to read from netCDF file",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=".",
        required=False,
        help="output directory for plots",
    )
    parser.add_argument(
        "-O",
        "--obsfile",
        type=str,
        nargs="+",
        required=False,
        help="File(s) containing obs data to compare against",
    )
    parser.add_argument(
        "-D",
        "--dataset",
        type=str,
        required=False,
        default="WOA13_annual_TS",
        help="Name of the observational dataset, \
              as provided in intake catalog",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode displays plot to screen. Default is False",
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=False,
        default="gfdl",
        help="computing platform, default is gfdl",
    )
    parser.add_argument(
        "-S",
        "--stream",
        type=str,
        required=False,
        help="stream output plot (diff/compare)",
    )
    cmdLineArgs = parser.parse_args(cliargs)
    return cmdLineArgs


def run(dictArgs):
    """ main can be called from either command line and then use parser from run()
    or DORA can build the args and run it directly """

    # read the data needed for plots
    x, y, area, model, obs = read(dictArgs)
    # make the plots
    plot(x, y, area, model, obs, dictArgs)


def plot(x, y, area, model, obs, dictArgs):
    """meta plotting function"""

    streamdiff = True if dictArgs["stream"] == "diff" else False
    streamcompare = True if dictArgs["stream"] == "compare" else False
    streamnone = True if dictArgs["stream"] is None else False

    # common plot properties
    pngout = dictArgs["outdir"]
    obsds = dictArgs["dataset"]
    var = dictArgs["var"]
    units = dictArgs["units"]
    clim_diff = dictArgs["clim_diff"]
    cmap_diff = dictArgs["cmap_diff"]
    clim_compare = dictArgs["clim_compare"]
    cmap_compare = dictArgs["cmap_compare"]

    if dictArgs["suptitle"] != "":
        suptitle = f"{dictArgs['suptitle']} {dictArgs['label']}"
    else:
        title = get_run_name(dictArgs["infile"])
        suptitle = f"{title} {dictArgs['label']}"

    title_diff = f"{var} bias (w.r.t. {obsds}) {units}"
    title1_compare = f"{var} {units}"
    title2_compare = f"{obsds} {var} {units}"
    png_diff = f"{pngout}/{var}_annual_bias_{obsds}.png"
    png_compare = f"{pngout}/{var}_annual_bias_{obsds}.3_panel.png"

    diff_kwargs = {
        "area": area,
        "suptitle": suptitle,
        "title": title_diff,
        "clim": clim_diff,
        "colormap": cmap_diff,
        "centerlabels": True,
        "extend": "both",
        "save": png_diff,
    }
    compare_kwargs = {
        "area": area,
        "suptitle": suptitle,
        "title1": title1_compare,
        "title2": title2_compare,
        "clim": clim_compare,
        "colormap": cmap_compare,
        "extend": "max",
        "dlim": clim_diff,
        "dcolormap": cmap_diff,
        "dextend": "both",
        "centerdlabels": True,
        "save": png_compare,
    }

    # make diff plot
    if streamdiff or streamnone:
        img = plot_xydiff(x, y, model, obs, diff_kwargs, stream=streamdiff)
        imgbufs = [img]

    # make compare plot
    if streamcompare or streamnone:
        img = plot_xycompare(x, y, model, obs, compare_kwargs, stream=streamcompare)
        imgbufs = [img]

    if not streamnone:
        return imgbufs


def parse_and_run(cliargs=None):
    cmdLineArgs = parse(cliargs)
    run(cmdLineArgs)


if __name__ == "__main__":
    parse_and_run()
