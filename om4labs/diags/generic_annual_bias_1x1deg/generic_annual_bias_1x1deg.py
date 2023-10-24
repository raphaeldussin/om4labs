#!/usr/bin/env python

import io
import numpy as np
import argparse
import xarray as xr
import warnings
import pkg_resources as pkgr

from om4labs import m6plot
from om4labs.helpers import get_run_name, try_variable_from_list
from om4labs.om4plotting import plot_xydiff, plot_xycompare
from om4labs.om4common import read_data, subset_data
from om4labs.om4common import simple_average, copy_coordinates
from om4labs.om4common import compute_area_regular_grid
from om4labs.om4common import date_range
from om4labs.om4common import image_handler
from om4labs.om4common import open_intake_catalog

from om4labs.om4parser import default_diag_parser

imgbufs = []


def read(dictArgs):
    """read data from model and obs files, process data and return it"""

    dsmodel = xr.open_mfdataset(
        dictArgs["infile"], combine="by_coords", use_cftime=True
    )

    if dictArgs["obsfile"] is not None:
        # priority to user-provided obs file
        dsobs = xr.open_mfdataset(
            dictArgs["obsfile"], combine="by_coords", decode_times=False
        )
    else:
        # use dataset from catalog, either from command line or default
        cat = open_intake_catalog(dictArgs["platform"], "obs")
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
    dataobs = copy_coordinates(datamodel, dataobs, ["assigned_lon", "assigned_lat"])

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

    # date range
    dates = date_range(dsmodel)

    return x, y, area, model, obs, dates


def parse(cliargs=None, template=False):
    """parse the command line arguments"""

    description = "Script for plotting annual-average bias to obs"
    parser = default_diag_parser(
        description=description, template=template, exclude=["basin", "topog"]
    )

    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        required=False,
        help="depth of field compared to obs",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="WOA13_annual_TS",
        help="Name of the observational dataset, \
              as provided in intake catalog",
    )

    parser.add_argument(
        "--style",
        type=str,
        required=False,
        default="diff",
        help="output plot style (diff/compare)",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def run(dictArgs):
    """main can be called from either command line and then use parser from run()
    or DORA can build the args and run it directly"""

    # read the data needed for plots
    x, y, area, model, obs, dates = read(dictArgs)
    # make the plots
    figs = plot(x, y, area, model, obs, dates, dictArgs)
    filename = f"{dictArgs['outdir']}/{dictArgs['var']}_{dictArgs['style']}"
    imgbufs = image_handler(figs, dictArgs, filename=filename)

    return imgbufs


def plot(x, y, area, model, obs, dates, dictArgs):
    """meta plotting function"""

    streamdiff = True if dictArgs["style"] == "diff" else False
    streamcompare = True if dictArgs["style"] == "compare" else False
    streamnone = True if dictArgs["style"] is None else False

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

    title_diff = f"{var} bias (w.r.t. {obsds}) {units}   Years {dates[0]} - {dates[1]}"
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

    figs = []

    # make diff plot
    if streamdiff or streamnone:
        fig = plot_xydiff(
            x,
            y,
            model,
            obs,
            diff_kwargs,
            interactive=dictArgs["interactive"],
            stream=streamdiff,
        )
        figs.append(fig)

    # make compare plot
    if streamcompare or streamnone:
        fig = plot_xycompare(
            x,
            y,
            model,
            obs,
            compare_kwargs,
            interactive=dictArgs["interactive"],
            stream=streamcompare,
        )
        figs.append(fig)

    return figs


def parse_and_run(cliargs=None):
    cmdLineArgs = parse(cliargs)
    imgbufs = run(cmdLineArgs)
    return imgbufs


if __name__ == "__main__":
    parse_and_run()
