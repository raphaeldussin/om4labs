#!/usr/bin/env python3

import argparse
import pkg_resources as pkgr
import intake
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from om4labs import m6plot
import palettable
import xarray as xr
import xoverturning
import warnings

from om4labs.om4common import horizontal_grid
from om4labs.om4common import read_topography
from om4labs.om4common import image_handler
from om4labs.om4common import is_symmetric
from om4labs.om4common import generate_basin_masks
from om4labs.om4common import date_range
from om4labs.om4parser import default_diag_parser

from om4labs.diags.moc.plotting import plot_z
from om4labs.diags.moc.plotting import plot_rho

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


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


def read(dictArgs, vcomp="vmo", ucomp="umo"):
    """Read required fields to plot MOC in om4labs

    Parameters
    ----------
    dictArgs : dict
        Dictionary containing argparse options
    vcomp : str, optional
        Name of meridional component of total residual
        freshwater transport, by default "vmo"
    ucomp : str, optional
        Name of zonal component of total residual
        freshwater transport, by default "umo"

    Returns
    -------
    xarray.DataSet
        Two xarray datasets; one containing `umo`, `vmo`
        and the other containing the grid information
    """

    # initialize an xarray.Dataset to hold the output
    dset = xr.Dataset()
    dset_grid = xr.Dataset()

    # read the infile and get u, v transport components
    infile = dictArgs["infile"]
    ds = xr.open_mfdataset(infile, combine="by_coords")
    dset["umo"] = ds[ucomp]
    dset["vmo"] = ds[vcomp]

    # detect symmetric grid
    outputgrid = "symetric" if is_symmetric(dset) else "nonsymetric"

    # determine vertical coordinate
    layer = "z_l" if "z_l" in ds.dims else "rho2_l" if "rho2_l" in ds.dims else None
    assert layer is not None, "Unrecognized vertical coordinate."

    # get vertical coordinate edges
    interface = "z_i" if layer == "z_l" else "rho2_i" if layer == "rho2_l" else None
    dset[interface] = ds[interface]

    # save layer and interface info for use later in the workflow
    dset.attrs["layer"] = layer
    dset.attrs["interface"] = interface

    # get horizontal v-cell grid info
    dsV = horizontal_grid(dictArgs, point_type="v", outputgrid=outputgrid)
    dset_grid["geolon_v"] = xr.DataArray(dsV.geolon.values, dims=("yq", "xh"))
    dset_grid["geolat_v"] = xr.DataArray(dsV.geolat.values, dims=("yq", "xh"))

    # get topography info
    _depth = read_topography(dictArgs, coords=ds.coords, point_type="t")
    _depth_v = read_topography(dictArgs, coords=ds.coords, point_type="v")

    # save bathymetry
    depth = np.where(np.isnan(_depth.to_masked_array()), 0.0, _depth)
    dset_grid["deptho"] = xr.DataArray(depth, dims=("yh", "xh"))

    # get the wet mask on the v-grid
    _wet_v = xr.where(_depth_v.isnull(), 0.0, 1.0)
    dset_grid["wet_v"] = xr.DataArray(_wet_v.values, dims=("yq", "xh"))

    # dset_grid requires `xq`
    dset_grid["xq"] = dset.xq

    # save date range as an attribute
    dates = date_range(ds)
    dset.attrs["dates"] = dates

    return (dset, dset_grid)


def calculate(dset, dset_grid):
    """Main calculation function

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset with grid values, umo, and vmo

    Returns
    -------
    xarray.DataArray
        Time-mean overturning streamfunction by basin 
    """

    layer = dset.layer
    interface = dset.interface

    # define list of basins
    basins = ["atl-arc", "indopac", "global"]

    # determine options to pass to xoverturning.calcmoc
    mask_output = True if layer == "z_l" else False
    vertical = "rho2" if layer == "rho2_l" else "z" if layer == "z_l" else None

    # iterate over basins
    otsfn = [
        xoverturning.calcmoc(
            dset,
            dset_grid,
            mask_output=mask_output,
            output_true_lat=True,
            basin=x,
            verbose=False,
            vertical=vertical,
        )
        for x in basins
    ]

    # combine into single DataArray
    otsfn = xr.concat(otsfn, dim="basin")
    otsfn = otsfn.transpose(otsfn.dims[1], otsfn.dims[0], ...)

    # take the time mean
    otsfn = otsfn.squeeze()
    if "time" in otsfn.dims:
        otsfn = otsfn.mean(dim="time")

    return otsfn


def plot(dset, otsfn, label=None):
    """Plotting wrapper that redirects to either the z-level
    or sigma plotting routine based on the data type

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset with grid values, umo, and vmo
    otsfn : xarray.DataArray
        Time-mean overturning streamfunction by basin 
    label : str, optional
        title/experiment name for plot labels, by default None

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle
    """
    layer = dset.layer

    fig = (
        # z-coordinate plot
        plot_z(
            otsfn.values,
            otsfn.lat.values,
            otsfn.z_i.values,
            dates=dset.dates,
            label=label,
        )
        if layer == "z_l"
        # sigma2-coordinate plot
        else plot_rho(
            otsfn.values,
            otsfn.lat.values,
            otsfn.rho2_i.values,
            dates=dset.dates,
            label=label,
        )
        if layer == "rho2_l"
        else None
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
    dset, dset_grid = read(dictArgs)

    # calculate otsfn
    otsfn = calculate(dset, dset_grid)

    # make the plots
    fig = plot(dset, otsfn, dictArgs["label"],)

    filename = f"{dictArgs['outdir']}/moc"
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
