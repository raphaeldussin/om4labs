#!/usr/bin/env python3

""" Generic Drift Plot Diagnostic """

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from omlabs.om4parser import default_diag_parser
from omlabs.om4common import image_handler


def parse(cliargs=None, template=False):
    """Function to capture the user-specified command line options

    Parameters
    ----------
    cliargs : argparse, optional
        Command line options from argparse, by default None
    template : bool, optional
        Return dictionary instead of parser, by default False

    Returns
    -------
        parsed command line arguments
    """

    description = "Generic diagnostic to plot a depth-vs-time drift time series"

    parser = default_diag_parser(
        description=description,
        template=template,
        exclude=[
            "suptitle",
            "gridspec",
            "static",
            "basin",
            "obsfile",
            "hgrid",
            "topog",
        ],
    )

    parser.add_argument(
        "--description",
        type=str,
        required=False,
        default="Layer-average drift vs. time",
        help="description string in subtitle",
    )
    parser.add_argument(
        "--varname",
        type=str,
        required=False,
        default=None,
        help="variable name to plot",
    )
    parser.add_argument(
        "--range",
        type=float,
        required=False,
        default=1.0,
        help="min/max data range for contouring",
    )
    parser.add_argument(
        "--interval", type=float, required=False, default=0.1, help="contour interval"
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs):
    """Read required fields

    Parameters
    ----------
    dictArgs : dict
        Dictionary containing argparse options

    Returns
    -------
    xarray.DataSet
    """

    dset = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)

    return dset


def calculate(dset, varname, tcoord="time"):
    """Function to calculate the drift time series

    This function calculates the drift of a given variable contained
    in a dataset as the anomaly relative to the first time level

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset
    varname : str
        Variable name to be analyzed
    tcoord : str, optional
        Time coordinate variable name, by default "time"

    Returns
    -------
    xarray.DataArray
        Anomaly time series
    """

    return dset[varname] - dset[varname].isel({tcoord: 0})


def plot(
    var,
    time=None,
    depth=None,
    tcoord="time",
    zcoord="z_l",
    label=None,
    vardesc=None,
    rangemax=1.0,
    interval=0.1,
    splitscale=2000.0,
):
    """Function to plot a depth vs. time contour figure

    Parameters
    ----------
    var : xarray.DataArray or np.Array
        Input data array in either xarray or numpy format
    time : np.Array, optional
        Time coordiate array in NumPy format, otherwise the array is
        obtained from the xarray variable coordinates, by default None
    depth : np.Array, optional
        Depth coordiate array in NumPy format, otherwise the array is
        obtained from the xarray variable coordinates, by default None
    tcoord : str, optional
        Name of xarray time coordinate, by default "time"
    zcoord : str, optional
        Name of xarray depth coordinate, by default "z_l"
    label : str, optional
        Experiment name label, by default None
    vardesc : str, optional
        Variable description string (subtitle), by default None
    rangemax : float, optional
        Plotting range defined by +/- `rangemax`, by default 1.
    interval : float, optional
        Contour interval, by default 0.1
    splitscale : float, optional
        Depth level that defines split vertical scale boundary,
        by default 2000.

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle
    """

    # get NumPy arrays from xarray
    if isinstance(var, xr.DataArray):
        time = var[tcoord].values
        depth = var[zcoord].values
        var = var.values

    # setup figure and axis
    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 1, 1)

    # setup contour range and intervals
    levels = np.arange(-1.0 * rangemax, rangemax, interval)
    cb = plt.contourf(time, depth, var.T, levels=levels, cmap="RdBu_r", extend="both")

    # add line contours and labels
    cs = plt.contour(time, depth, var.T, levels=levels, colors=["k"], linewidths=0.5)
    ax.clabel(cs, fontsize=6, rightside_up=True)

    # add vertical line at split-scale location
    _ = plt.axhline(splitscale, color="k", linestyle="dashed", linewidth=0.75)
    ax.set_yscale("splitscale", zval=[6500.0, splitscale, 0.0])

    # add colorbar
    cb = plt.colorbar(cb)

    # set tick label sizes
    ax.tick_params(labelsize=8)
    cb.ax.tick_params(labelsize=8)

    # add titles
    ax.text(0.0, 1.1, label, transform=ax.transAxes, weight="bold", fontsize=8)
    ax.text(0.0, 1.04, vardesc, transform=ax.transAxes, style="italic", fontsize=7)

    # add min/max values
    maxval = str(round(var.max(), 2))
    minval = str(round(var.min(), 2))
    ax.text(
        1.0,
        1.04,
        f"Min: {minval}",
        ha="right",
        transform=ax.transAxes,
        style="italic",
        fontsize=7,
    )
    ax.text(
        1.0,
        1.09,
        f"Max: {maxval}",
        ha="right",
        transform=ax.transAxes,
        style="italic",
        fontsize=7,
    )

    return fig


def run(dictArgs):
    """Function to call read, calc, and plot in sequence

    Parameters
    ----------
    dictArgs : dict
        Dictionary of parsed options

    Returns
    -------
    io.BytesIO
        In-memory image buffer
    """

    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")

    # read in data
    dset = read(dictArgs)

    # calculate
    arr = calculate(dset, dictArgs["varname"])

    # make the plots
    fig = plot(
        arr,
        label=dictArgs["label"],
        vardesc=dictArgs["description"],
        rangemax=dictArgs["range"],
        interval=dictArgs["interval"],
    )

    filename = f"{dictArgs['outdir']}/drift_{dictArgs['varname']}"
    imgbufs = image_handler([fig], dictArgs, filename=filename)

    return imgbufs


def parse_and_run(cliargs=None):
    """Parses command line and runs diagnostic

    Parameters
    ----------
    cliargs : argparse, optional
        command line arguments from upstream instance, by default None

    Returns
    -------
    io.BytesIO
        In-memory image buffer
    """
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
