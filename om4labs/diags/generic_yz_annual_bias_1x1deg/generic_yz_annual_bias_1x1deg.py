#!/usr/bin/env python

import io
import numpy as np
import argparse
import xarray as xr
import warnings
import pkg_resources as pkgr
import intake

from om4labs import m6plot
from om4labs.helpers import get_run_name, try_variable_from_list
from om4labs.om4plotting import plot_yzdiff, plot_yzcompare
from om4labs.om4common import read_data, subset_data
from om4labs.om4common import simple_average, copy_coordinates
from om4labs.om4common import compute_area_regular_grid
from om4labs.om4common import DefaultDictParser
from om4labs.om4common import image_handler

from cmip_basins import generate_basin_codes

imgbufs = []


def read(dictArgs):
    """ read data from model and obs files, process data and return it """

    if dictArgs["model"] is not None:
        # use dataset from catalog, either from command line or default
        cat_platform = (
            f"catalogs/{dictArgs['model']}_catalog_{dictArgs['platform']}.yml"
        )
        catfile = pkgr.resource_filename("om4labs", cat_platform)
        cat = intake.open_catalog(catfile)
        ds_static = cat["ocean_static_1x1"].to_dask()

    if dictArgs["static"] is not None:
        ds_static = xr.open_dataset(dictArgs["static"])

    # Compute basin codes
    codes = generate_basin_codes(ds_static, lon="lon", lat="lat")
    codes = np.array(codes)

    # depth coordinate
    if "deptho" in list(ds_static.variables):
        depth = ds_static.deptho.to_masked_array()
    elif "depth" in list(ds_static.variables):
        depth = ds_static.depth.to_masked_array()
    else:
        raise ValueError("Unable to find depth field.")
    depth = np.where(np.isnan(depth), 0.0, depth)
    depth = depth * -1.0

    dsmodel = xr.open_mfdataset(
        dictArgs["infile"], combine="by_coords", decode_times=False
    )

    if dictArgs["obsfile"] is not None:
        # priority to user-provided obs file
        dsobs = xr.open_mfdataset(
            dictArgs["obsfile"], combine="by_coords", decode_times=False
        )
    else:
        # use dataset from catalog, either from command line or default
        cat_platform = "catalogs/obs_catalog_" + dictArgs["platform"] + ".yml"
        catfile = pkgr.resource_filename("om4labs", cat_platform)
        cat = intake.open_catalog(catfile)
        dsobs = cat[dictArgs["dataset"]].to_dask()

    # read in model and obs data
    datamodel = read_data(dsmodel, dictArgs["possible_variable_names"])
    dataobs = read_data(dsobs, dictArgs["possible_variable_names"])

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

    # check final data is 3d
    assert len(datamodel.dims) == 3
    assert len(dataobs.dims) == 3

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
    z = datamodel["assigned_depth"].values

    # convert z to negative values
    z = z * -1

    # compute area
    if "areacello" in dsmodel.variables:
        area = dsmodel["areacello"].values
    else:
        if (model.shape[-2], model.shape[-1]) == (180, 360):
            area = compute_area_regular_grid(dsmodel)
        else:
            raise IOError("no cell area provided")

    return y, z, depth, area, codes, model, obs


def parse(cliargs=None, template=False):
    """ parse the command line arguments """

    if template is True:
        parser = DefaultDictParser(
            description="Script for plotting \
                                                      annual-average bias to obs"
        )
    else:
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

    parser.add_argument("-m", "--model", type=str, default=None, help="Model Class")

    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=".",
        required=False,
        help="output directory for plots",
    )
    parser.add_argument(
        "-F",
        "--format",
        type=str,
        default="png",
        required=False,
        help="output format type",
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
        "--style",
        type=str,
        required=False,
        default="diff",
        help="output plot style (diff/compare)",
    )
    parser.add_argument(
        "--static", "--static", type=str, default=None, help="Path to static file"
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def run(dictArgs):
    """main can be called from either command line and then use parser from run()
    or DORA can build the args and run it directly"""

    # read the data needed for plots
    y, z, depth, area, code, model, obs = read(dictArgs)

    if len(code.shape) == 2:
        code = np.tile(code[None, :, :], (len(z), 1, 1))

    figs = []
    filename = []

    # global
    figs.append(
        plot(y, z, depth, area, model.mean(axis=-1), obs.mean(axis=-1), dictArgs)[0]
    )
    filename.append(
        f"{dictArgs['outdir']}/{dictArgs['var']}_yz_gbl_{dictArgs['style']}"
    )

    # atlantic
    mask = code * 0
    mask[(code == 2) | (code == 4)] = 1
    _model = np.ma.masked_where(mask == 0, model).mean(axis=-1)
    _obs = np.ma.masked_where(mask == 0, obs).mean(axis=-1)
    _depth = np.ma.masked_where(mask[0] == 0, depth)
    _area = np.ma.masked_where(mask[0] == 0, area)
    figs.append(plot(y, z, _depth, _area, _model, _obs, dictArgs)[0])
    filename.append(
        f"{dictArgs['outdir']}/{dictArgs['var']}_yz_atl_{dictArgs['style']}"
    )

    # indopacific
    mask = code * 0
    mask[(code == 3) | (code == 5)] = 1
    _model = np.ma.masked_where(mask == 0, model).mean(axis=-1)
    _obs = np.ma.masked_where(mask == 0, obs).mean(axis=-1)
    _depth = np.ma.masked_where(mask[0] == 0, depth)
    _area = np.ma.masked_where(mask[0] == 0, area)
    figs.append(plot(y, z, _depth, _area, _model, _obs, dictArgs)[0])
    filename.append(
        f"{dictArgs['outdir']}/{dictArgs['var']}_yz_pac_{dictArgs['style']}"
    )

    imgbufs = image_handler(figs, dictArgs, filename=filename)

    return imgbufs


def plot(y, z, depth, area, model, obs, dictArgs):
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

    title_diff = f"{var} bias (w.r.t. {obsds}) {units}"
    title1_compare = f"{var} {units}"
    title2_compare = f"{obsds} {var} {units}"
    png_diff = f"{pngout}/{var}_yz_annual_bias_{obsds}.png"
    png_compare = f"{pngout}/{var}_yz_annual_bias_{obsds}.3_panel.png"

    diff_kwargs = {
        "area": area,
        "depth": depth,
        "suptitle": suptitle,
        "title": title_diff,
        "clim": clim_diff,
        "colormap": cmap_diff,
        "splitscale": [0, -2000, -6500],
        "centerlabels": True,
        "extend": "both",
        "save": png_diff,
    }
    compare_kwargs = {
        "area": area,
        "depth": depth,
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
        fig = plot_yzdiff(
            y,
            z,
            model,
            obs,
            diff_kwargs,
            interactive=dictArgs["interactive"],
            stream=streamdiff,
        )
        figs.append(fig)

    # make compare plot
    if streamcompare or streamnone:
        fig = plot_yzcompare(
            y,
            z,
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
