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
from om4labs.om4common import generate_basin_masks
from om4labs.om4common import date_range
from om4labs.om4parser import default_diag_parser

from om4labs.diags.moc.plotting import plot_z
from om4labs.diags.moc.plotting import plot_rho

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


def parse(cliargs=None, template=False):
    """
    Function to capture the user-specified command line options
    """
    description = """ """

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
        Xarray dataset containing `umo`, `vmo`, `geolon`, 
        `geolat`, `depth`, `zmod`, `wet`, and `basin_masks`
    """

    # initialize an xarray.Dataset to hold the output
    dset = xr.Dataset()

    # read the infile and get u, v transport components
    infile = dictArgs["infile"]
    ds = xr.open_mfdataset(infile, combine="by_coords")
    dset["umo"] = ds[ucomp]
    dset["vmo"] = ds[vcomp]

    # determine vertical coordinate
    layer = "z_l" if "z_l" in ds.dims else "rho2_l" if "rho2_l" in ds.dims else None
    assert layer is not None, "Unrecognized vertical coordinate."

    # get vertical coordinate edges
    interface = "z_i" if layer == "z_l" else "rho2_i" if layer == "rho2_l" else None
    dset[interface] = ds[interface]

    # save layer and interface info for use later in the workflow
    dset.attrs["layer"] = layer
    dset.attrs["interface"] = interface

    # get horizontal t-cell grid info
    dsT = horizontal_grid(dictArgs, point_type="t")
    dset["geolon"] = xr.DataArray(dsT.geolon.values, dims=("yh", "xh"))
    dset["geolat"] = xr.DataArray(dsT.geolat.values, dims=("yh", "xh"))

    # get topography info
    _depth = read_topography(dictArgs, coords=ds.coords, point_type="t")
    depth = np.where(np.isnan(_depth.to_masked_array()), 0.0, _depth)
    dset["depth"] = xr.DataArray(depth, dims=("yh", "xh"))

    # replicates older get_z() func from m6plot
    zmod, _ = xr.broadcast(dset[interface], xr.ones_like(dset.geolat))
    zmod = xr.ufuncs.minimum(dset.depth, xr.ufuncs.fabs(zmod)) * -1.0
    zmod = zmod.transpose(interface, "yh", "xh")
    dset["zmod"] = zmod

    # grid wet mask based on model's topography
    wet = np.where(np.isnan(_depth.to_masked_array()), 0.0, 1.0)
    dset["wet"] = xr.DataArray(wet, dims=("yh", "xh"))

    # basin masks
    basin_code = dsT.basin.values
    basins = ["atlantic_arctic", "indo_pacific"]
    basins = [generate_basin_masks(basin_code, basin=x) for x in basins]
    basins = [xr.DataArray(x, dims=("yh", "xh")) for x in basins]
    basins = xr.concat(basins, dim="basin")
    dset["basin_masks"] = basins

    # date range
    dates = date_range(ds)
    dset.attrs["dates"] = dates

    return dset


def calculate(dset):
    """Main computational script"""

    basins = ["atl-arc", "indopac", "global"]
    otsfn = [
        xoverturning.calcmoc(
            dset, basin=x, layer=dset.layer, interface=dset.interface, verbose=False
        )
        for x in basins
    ]
    otsfn = xr.concat(otsfn, dim="basin")
    otsfn = otsfn.transpose(otsfn.dims[1], otsfn.dims[0], ...)

    return otsfn


def plot(dset, otsfn, label=None):
    layer = dset.layer

    fig = (
        # z-coordinate plot
        plot_z(dset, otsfn, label=label)
        if layer == "z_l"
        # sigma2-coordinate plot
        else plot_rho(dset, otsfn, label=label)
        if layer == "rho2_l"
        else None
    )

    return fig


def run(dictArgs):
    """Function to call read, calc, and plot in sequence"""

    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")

    # read in data
    dset = read(dictArgs)

    # calculate otsfn
    otsfn = calculate(dset)

    # make the plots
    fig = plot(dset, otsfn, dictArgs["label"],)
    # ---------------------

    filename = f"{dictArgs['outdir']}/moc"
    imgbufs = image_handler([fig], dictArgs, filename=filename)

    return imgbufs


def parse_and_run(cliargs=None):
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
