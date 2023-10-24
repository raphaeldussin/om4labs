#!/usr/bin/env python3

import pkg_resources as pkgr
import intake
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from omlabs import m6plot
import palettable
import xarray as xr
import xoverturning
import warnings

# these are always needed
from omlabs.om4parser import default_diag_parser
from omlabs.om4common import image_handler

# these are available to you to use
from omlabs.om4common import date_range
from omlabs.om4common import generate_basin_masks
from omlabs.om4common import horizontal_grid
from omlabs.om4common import is_symmetric
from omlabs.om4common import read_topography


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

    description = " "

    parser = default_diag_parser(description=description, template=template)

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
    dset_static = xr.open_dataset(dictArgs["static"])

    return (dset, dset_static)


def calculate(dset, dset_static):
    """Main calculation function

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset
    dset_static : xarray.Dataset
        Input static dataset

    Returns
    -------
    xarray.DataArray
    """

    return xr.DataArray()


def plot(arr, label=None):
    """Plotting wrapper

    Parameters
    ----------
    arr : xarray.DataArray
        Input data array

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle
    """

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    ax.plot(arr.values)

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
    dset, dset_static = read(dictArgs)

    # calculate
    arr = calculate(dset, dset_static)

    # make the plots
    fig = plot(arr, dictArgs["label"])

    filename = f"{dictArgs['outdir']}/template"
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
