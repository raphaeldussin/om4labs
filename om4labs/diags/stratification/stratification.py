#!/usr/bin/env python3

import pkg_resources as pkgr
import intake
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from om4labs import m6plot
import palettable
import xarray as xr
import xcompare
import xoverturning
import warnings

# these are always needed
from om4labs.om4parser import default_diag_parser
from om4labs.om4common import image_handler

# these are available to you to use
from om4labs.om4common import date_range
from om4labs.om4common import generate_basin_masks
from om4labs.om4common import horizontal_grid
from om4labs.om4common import is_symmetric
from om4labs.om4common import open_intake_catalog
from om4labs.om4common import read_topography


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

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="WOA13_annual_TS",
        help="Name of the observational dataset, \
              as provided in intake catalog",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs, infile=None):
    """Read required fields

    Parameters
    ----------
    dictArgs : dict
        Dictionary containing argparse options

    Returns
    -------
    xarray.DataSet
    """

    if dictArgs["obsfile"] is not None:
        # priority to user-provided obs file
        dsobs = xr.open_mfdataset(
            dictArgs["obsfile"], combine="by_coords", decode_times=False
        )
    else:
        # use dataset from catalog, either from command line or default
        cat = open_intake_catalog(dictArgs["platform"], "obs")
        dsobs = cat[dictArgs["dataset"]].to_dask()

    obs = dsobs["ptemp"]

    infile = dictArgs["infile"] if infile is None else infile
    model = xr.open_mfdataset(infile, use_cftime=True)
    model = model["thetao"].mean(dim="time")

    obs = obs.squeeze()
    model = model.squeeze()

    obs = obs.reset_coords(drop=True)
    model = model.reset_coords(drop=True)

    return model, obs


def calculate(model, obs, zcoord="z_l", depth_range=(25, 1000)):
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

    dtdz = model.diff(zcoord).sel({zcoord: slice(*depth_range)})
    dtdz.load()
    mask = xr.where(dtdz.isel({zcoord: 0}).isnull(), np.nan, 1.0)
    dtdz = dtdz.fillna(0.0)
    dtdz_max = dtdz.min(zcoord)
    dtdz_max_depth = xr.broadcast(dtdz.z_l, dtdz)[0][dtdz.argmin(zcoord)]
    model = xr.Dataset(
        {"dtdz_max": dtdz_max * mask, "dtdz_max_depth": dtdz_max_depth * mask}
    )

    dtdz = obs.diff(zcoord).sel({zcoord: slice(*depth_range)})
    dtdz.load()
    mask = xr.where(dtdz.isel({zcoord: 0}).isnull(), np.nan, 1.0)
    dtdz = dtdz.fillna(0.0)
    dtdz_max = dtdz.min(zcoord)
    dtdz_max_depth = xr.broadcast(dtdz.z_l, dtdz)[0][dtdz.argmin(zcoord)]
    obs = xr.Dataset(
        {"dtdz_max": dtdz_max * mask, "dtdz_max_depth": dtdz_max_depth * mask}
    )

    return xcompare.compare_datasets(model, obs)


def plot(results):
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

    fig = [
        xcompare.plot_three_panel(results, "dtdz_max"),
        xcompare.plot_three_panel(results, "dtdz_max_depth"),
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
    model, obs = read(dictArgs)

    # calculate
    results = calculate(model, obs)

    # make the plots
    fig = plot(results)

    filename = [
        f"{dictArgs['outdir']}/thermocline_strength",
        f"{dictArgs['outdir']}/thermocline_depth",
    ]
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
