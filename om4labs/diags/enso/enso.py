#!/usr/bin/env python3

import pkg_resources as pkgr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xwavelet as xw
import warnings

from om4labs.om4common import image_handler
from om4labs.om4common import open_intake_catalog
from om4labs.om4parser import default_diag_parser

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")

# define NINO regions

nino12 = {
    "lat_range": (-10, 0.0),
    "lon_range": (270.0, 280.0),
    "label": 1.2,
}

nino3 = {
    "lat_range": (-5.0, 5.0),
    "lon_range": (210.0, 270.0),
    "label": 3,
}
nino34 = {
    "lat_range": (-5.0, 5.0),
    "lon_range": (190.0, 240.0),
    "label": 3.4,
}
nino4 = {
    "lat_range": (-5.0, 5.0),
    "lon_range": (160.0, 210.0),
    "label": 4,
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

    exclude = ["basin", "hgrid", "topog", "gridspec", "config"]

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

    # Open model array and static file
    array = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)[dictArgs["varname"]]
    areacello = xr.open_dataset(dictArgs["static"])["areacello"].fillna(0.0)

    # Open pre-calculated ENSO spectra from Obs.
    if dictArgs["obsfile"] is not None:
        ref1 = xr.open_dataset(dictArgs["obsfile"])
        ref1 = ref1["spectrum"]
        ref1.attrs = {**ref1.attrs, "label": "Reference"}
        reference = [ref1]

    else:
        cat = open_intake_catalog(dictArgs["platform"], "obs")

        # open reference datasets
        ref1 = cat["wavelet_NOAA_ERSST_v5_1957_2002"].to_dask()
        ref2 = cat["wavelet_NOAA_ERSST_v5_1880_2019"].to_dask()

        # select spectrum variable
        ref1 = ref1["spectrum"]
        ref2 = ref2["spectrum"]

        # set attributes
        ref1.attrs = {**ref1.attrs, "label": "ERSST v5 1957-2002"}
        ref2.attrs = {**ref2.attrs, "label": "ERSST v5 1880-2019"}

        reference = [ref1, ref2]

    return (array, areacello, reference)


def calculate(array, areacello, timedim="time", reference=None):
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

    if reference is not None:
        reference = [reference] if not isinstance(reference, list) else reference

    # get a set of variable dimentions excluding "time"
    dims = list(array.dims)
    dims.remove(timedim)
    dims = tuple(dims)

    latdim = "lat"
    londim = "lon"

    array = [
        (
            array.sel({latdim: slice(*x["lat_range"]), londim: slice(*x["lon_range"])})
            .weighted(
                areacello.sel(
                    {latdim: slice(*x["lat_range"]), londim: slice(*x["lon_range"])}
                )
            )
            .mean(dims)
            .assign_attrs({**attrs, "region": f"NINO{x['label']}"}),
            [y.sel(region=x["label"]) for y in reference],
        )
        for x in regions
    ]

    return [xw.Wavelet(x[0], reference=x[1], detrend=False, scaled=True) for x in array]


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
    array, areacello, reference = read(dictArgs)

    # calculate otsfn
    results = calculate(array, areacello, reference=reference)

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
