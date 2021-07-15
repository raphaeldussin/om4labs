#!/usr/bin/env python3

import pkg_resources as pkgr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xwavelet as xw
import warnings

from om4labs.om4common import image_handler
from om4labs.om4parser import default_diag_parser

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")

# define NINO regions

nino12 = {
    "lat_range": (-10, 0.0),
    "lon_range": (270.0, 280.0),
    "label": "NINO1+2",
}

nino3 = {
    "lat_range": (-5.0, 5.0),
    "lon_range": (210.0, 270.0),
    "label": "NINO3",
}
nino34 = {
    "lat_range": (-5.0, 5.0),
    "lon_range": (190.0, 240.0),
    "label": "NINO3+4",
}
nino4 = {
    "lat_range": (-5.0, 5.0),
    "lon_range": (160.0, 210.0),
    "label": "NINO4",
}

regions = [nino12, nino3, nino34, nino4]


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

    exclude = ["platform", "basin", "obsfile", "hgrid", "topog", "gridspec", "config"]

    parser = default_diag_parser(
        description=description, template=template, exclude=exclude
    )

    parser.add_argument(
        "--varname",
        type=str,
        default="tos",
        help="Variable to analyze. Default is tos.",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs):
    """Read required fields for ENSO analysis

    Parameters
    ----------
    dictArgs : dict
        Dictionary containing argparse options

    Returns
    -------
    (xarray.DataArray, xarray.DataArray)
    """

    array = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)[dictArgs["varname"]]
    areacello = xr.open_dataset(dictArgs["static"])["areacello"].fillna(0.0)

    array.load()

    return (array, areacello)


def calculate(array, areacello, timedim="time"):
    """Main calculation function

    Parameters
    ----------
    array : xarray.DataArray
        Time series data array
    areacello : xarray.DataArray
        Cell area data array
    timedim : str, optional
        Name of time dimension, by default "time"

    Returns
    -------
    [wavelet.Wavelet]
    """

    attrs = array.attrs

    # get a set of variable dimentions excluding "time"
    dims = list(array.dims)
    dims.remove(timedim)
    dims = tuple(dims)

    latdim = "lat"
    londim = "lon"

    array = [
        array.sel({latdim: slice(*x["lat_range"]), londim: slice(*x["lon_range"])})
        .weighted(
            areacello.sel(
                {latdim: slice(*x["lat_range"]), londim: slice(*x["lon_range"])}
            )
        )
        .mean(dims)
        .assign_attrs({**attrs, "region": x["label"]})
        for x in regions
    ]

    return [xw.Wavelet(x, detrend=False, scaled=True) for x in array]


def plot(results, label):
    """Plotting wrapper for composite plots

    Parameters
    ----------
    results : list of wavelet.Wavelet
        Wavelet class objects
    label : str
        Experiment name

    Returns
    -------
    [matplotlib.figure.Figure]
    """
    fig = [
        x.composite(title=f"ENSO Analysis - {x.dset.timeseries.region}", subtitle=label)
        for x in results
    ]

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
    array, areacello = read(dictArgs)

    # calculate otsfn
    results = calculate(array, areacello)

    # make the plots
    fig = plot(results, dictArgs["label"])

    filename = [f"{dictArgs['outdir']}/enso_{x['label']}" for x in regions]
    imgbufs = image_handler(fig, dictArgs, filename=filename)

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
