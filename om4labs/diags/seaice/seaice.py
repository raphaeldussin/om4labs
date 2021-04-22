#!/usr/bin/env python3

"""
om4labs: model-simulated sea ice vs. NSIDC obs
"""

__all__ = ["parse", "read", "calculate", "plot", "run", "parse_and_run"]

import argparse
import copy
import time
import warnings

import cartopy.crs as ccrs
import cartopy.feature
import intake
import matplotlib as mpl
import matplotlib.pyplot as plt
import pkg_resources as pkgr
from matplotlib.lines import Line2D

from om4labs.om4common import (
    annual_cycle,
    curv_to_curv,
    date_range,
    image_handler,
    open_intake_catalog,
    standard_grid_cell_area,
)
from om4labs.om4parser import default_diag_parser

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")

import io
import os
import shutil
import tempfile

import numpy as np
import palettable
import xarray as xr


def parse(cliargs=None, template=False):
    description = """Plot sea ice vs. NSIDC"""

    parser = default_diag_parser(
        description=description, template=template, exclude=["basin", "topog"]
    )

    parser.add_argument(
        "--month", type=str, default="March", help="Month to analyze. Deafult is March",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="nh",
        help="Region for analysis. Default is nh. Options are nh and sh",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


# def read(infile, obsfile, static):
def read(dictArgs):
    """Function to read in the data. Returns xarray datasets"""

    infile = dictArgs["infile"]

    # Open ice model output and the static file
    ds = xr.open_mfdataset(infile, combine="by_coords")
    if "siconc" in ds.variables:
        ds["CN"] = ds["siconc"]
        if ds["CN"].max() > 1.0:
            ds["CN"] = ds["CN"] / 100.0
    else:
        ds["CN"] = ds["CN"].sum(dim="ct")

    # Detect if we are native grid or 1x1
    if (ds.CN.shape[-2] == 180) and (ds.CN.shape[-1] == 360):
        standard_grid = True
    else:
        standard_grid = False

    if dictArgs["config"] is not None:
        # use dataset from catalog, either from command line or default
        cat_platform = (
            f"catalogs/{dictArgs['config']}_catalog_{dictArgs['platform']}.yml"
        )
        catfile = pkgr.resource_filename("om4labs", cat_platform)
        cat = intake.open_catalog(catfile)
        if standard_grid is True:
            dstatic = cat["ice_static_1x1"].to_dask()
        else:
            dstatic = cat["ice_static"].to_dask()

    # Override static file if provided
    if dictArgs["static"] is not None:
        dstatic = xr.open_dataset(dictArgs["static"])

    # Append static fields to the return Dataset
    if standard_grid is True:
        _lon = np.array(dstatic["lon"].to_masked_array())
        _lat = np.array(dstatic["lat"].to_masked_array())
        X, Y = np.meshgrid(_lon, _lat)
        ds["GEOLON"] = xr.DataArray(X, dims=["lat", "lon"])
        ds["GEOLAT"] = xr.DataArray(Y, dims=["lat", "lon"])
        _AREA = standard_grid_cell_area(_lat, _lon)
        _MASK = np.array(dstatic["mask"].fillna(0.0).to_masked_array())
        _AREA = _AREA * _MASK
        ds["CELL_AREA"] = xr.DataArray(_AREA, dims=["lat", "lon"])
        ds["AREA"] = xr.DataArray(_AREA, dims=["lat", "lon"])
        ds = ds.rename({"lon": "x", "lat": "y"})
    else:
        ds["CELL_AREA"] = dstatic["CELL_AREA"]
        ds["GEOLON"] = dstatic["GEOLON"]
        ds["GEOLAT"] = dstatic["GEOLAT"]
        ds["AREA"] = dstatic["CELL_AREA"] * 4.0 * np.pi * (6.378e6 ** 2)

    # Get Valid Mask
    valid_mask = np.where(ds["CELL_AREA"] == 0.0, True, False)

    # Open observed SIC on 25-km EASE grid (coords already named lat and lon)
    if dictArgs["obsfile"] is not None:
        dobs = xr.open_dataset(dictArgs["obsfile"])
    else:
        cat = open_intake_catalog(dictArgs["platform"], "obs")
        dobs = cat[f"NSIDC_{dictArgs['region'].upper()}_monthly"].to_dask()

    # Close the static file (no longer used)
    dstatic.close()

    return ds, dobs, valid_mask


def calculate(ds, dobs, region="nh"):
    """ Function to calculate sea ice parameters """

    # Container dictionaries to hold results
    model = xr.Dataset()
    obs = xr.Dataset()

    # Add coordinates
    model["GEOLON"] = ds["GEOLON"]
    model["GEOLAT"] = ds["GEOLAT"]
    model = model.rename({"GEOLON": "lon", "GEOLAT": "lat"})
    obs["lon"] = dobs["lon"]
    obs["lat"] = dobs["lat"]

    # Create annual cycle climatology
    model["ac"] = annual_cycle(ds, "CN")
    obs["ac"] = annual_cycle(dobs, "sic")

    # Regrid the observations to the model grid (for plotting)
    regridded = curv_to_curv(obs, model, reuse_weights=False)

    # Calculate area and extent
    if region == "nh":
        model["area"] = xr.where(ds["GEOLAT"] > 0.0, model["ac"] * ds.AREA, 0.0)
        model["ext"] = xr.where(
            (model["ac"] > 0.15) & (ds["GEOLAT"] > 0.0), ds.AREA, 0.0
        )
    elif region == "sh":
        model["area"] = xr.where(ds["GEOLAT"] < 0.0, model["ac"] * ds.AREA, 0.0)
        model["ext"] = xr.where(
            (model["ac"] > 0.15) & (ds["GEOLAT"] < 0.0), ds.AREA, 0.0
        )
    else:
        raise ValueError(f"Unknown region {region}. Option are nh or sh")

    # Ensure dims are in the correct order
    model["area"] = model["area"].transpose("month", ...)
    model["ext"] = model["ext"].transpose("month", ...)

    # Sum to get model area and extent
    model["area"] = model["area"].sum(axis=(-2, -1)) * 1.0e-12
    model["ext"] = model["ext"].sum(axis=(-2, -1)) * 1.0e-12

    # Get obs model and extent
    obs["area"] = obs["ac"] * dobs.areacello
    obs["area"] = obs["area"].transpose("month", ...)
    obs["area"] = obs["area"].sum(axis=(-2, -1)) * 1.0e-12

    obs["ext"] = xr.where(obs["ac"] > 0.15, dobs.areacello, 0.0)
    obs["ext"] = obs["ext"].transpose("month", ...)
    obs["ext"] = obs["ext"].sum(axis=(-2, -1)) * 1.0e-12

    # Get tuple of start year and end years for model and observations
    model.attrs["time"] = date_range(ds)
    obs.attrs["time"] = date_range(dobs)

    return model, obs, regridded


def _plot_annual_cycle(ax, _mod, _obs, roll=0):
    """Creates an anual cycle subplot panel"""
    _months = np.array(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

    # Shift annual cycle
    _months = np.roll(_months, roll)
    _obs = np.roll(_obs, roll)
    _mod = np.roll(_mod, roll)

    # Add wrap-around point
    _months = np.append(_months, _months[0])
    _obs = np.append(_obs, _obs[0])
    _mod = np.append(_mod, _mod[0])

    # Plot the data
    t = np.arange(0, len(_mod))
    ax.plot(t, _mod, "r", label="model")
    ax.plot(t, _obs, "k", label="obs")

    # Format the plot
    ax.set_xticks(np.arange(0, len(_obs)))
    ax.set_xticklabels(_months)
    plt.legend()


def _plot_map_panel(
    ax,
    x,
    y,
    plotdata,
    cmap="Blues_r",
    vmin=0.01,
    vmax=100.0,
    extent=[-180, 180, 50, 90],
    contour=True,
):
    """Function to plot a map-based sea ice panel"""
    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    cmap.set_bad(color="#555555", alpha=1)
    ax.set_extent(extent, ccrs.PlateCarree())
    cb = ax.pcolormesh(
        x, y, plotdata, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax
    )
    if contour is True:
        cs2 = ax.contour(
            x, y, plotdata, transform=ccrs.PlateCarree(), levels=[15.0], colors=["r"]
        )
    return cb


def plot(model, obs, regridded, valid_mask, label=None, region="nh", month="March"):
    """Function to make sea ice plot"""

    # Get integer index of the requested month
    month_index = int(time.strptime(month, "%B").tm_mon) - 1

    # Setup figure canvas
    fig = plt.figure(figsize=(11, 8.5))

    # Define projection
    if region == "nh":
        proj = ccrs.NorthPolarStereo(central_longitude=-100)
        extent = [-180, 180, 50, 90]
    elif region == "sh":
        proj = ccrs.SouthPolarStereo(central_longitude=-100)
        extent = [-180, 180, -50, -90]
    else:
        raise ValueError(f"Unknown region {region}. Option are nh or sh")

    # All maps are plotted on the model grid
    x = np.array(model["lon"].to_masked_array())
    y = np.array(model["lat"].to_masked_array())

    # Top left panel - model map of sea ice
    ax = plt.subplot(2, 3, 1, projection=proj)
    plotdata = (model["ac"][month_index] * 100.0).to_masked_array()
    plotdata = np.ma.masked_where(valid_mask, plotdata)
    cb1 = _plot_map_panel(ax, x, y, plotdata, extent=extent)
    ax.set_title(f"Model - Years {model.time[0]} to {model.time[1]}")
    fig.colorbar(
        cb1, orientation="horizontal", fraction=0.03, pad=0.05, aspect=60, ax=ax
    )

    # Top middle panel - observed map of sea ice
    ax = plt.subplot(2, 3, 2, projection=proj)
    plotdata = (regridded["ac"][month_index] * 100.0).to_masked_array()
    plotdata = np.ma.masked_where(valid_mask, plotdata)
    cb2 = _plot_map_panel(ax, x, y, plotdata, extent=extent)
    ax.set_title(f"NSIDC - Years {obs.time[0]} to {obs.time[1]}")
    fig.colorbar(
        cb2, orientation="horizontal", fraction=0.03, pad=0.05, aspect=60, ax=ax
    )

    # Top right panel - model minus observed difference
    ax = plt.subplot(2, 3, 3, projection=proj)
    _mod = np.where(
        np.isnan(model["ac"][month_index].data), 0.0, model["ac"][month_index].data
    )
    _obs = np.where(
        np.isnan(regridded["ac"][month_index].data),
        0.0,
        regridded["ac"][month_index].data,
    )
    plotdata = (_mod - _obs) * 100.0
    plotdata = np.ma.masked_where(valid_mask, plotdata)
    cb3 = _plot_map_panel(
        ax,
        x,
        y,
        plotdata,
        cmap="RdBu",
        vmin=-100.0,
        vmax=100.0,
        contour=False,
        extent=extent,
    )
    ax.set_title("Difference")
    fig.colorbar(
        cb3, orientation="horizontal", fraction=0.03, pad=0.05, aspect=60, ax=ax
    )

    # Bottom left panel - annual cycle of sea ice area
    ax = plt.subplot(2, 3, 4)
    _plot_annual_cycle(ax, model["area"], obs["area"], roll=month_index - 9)
    ax.set_title("Sea Ice Area")
    ax.set_ylabel("1.e6 km^2")

    # Bottom middle panel - annual cycle of sea ice extent
    ax = plt.subplot(2, 3, 5)
    _plot_annual_cycle(ax, model["ext"], obs["ext"], roll=month_index - 9)
    ax.set_title("Sea Ice Extent")
    ax.set_ylabel("1.e6 km^2")

    # Text statistics annotations
    fig.text(0.67, 0.39, "Annual Sea Ice Area", fontsize=10)
    fig.text(0.67, 0.375, "Model Max: %0.5f" % (model["area"].max()), fontsize=10)
    fig.text(0.67, 0.36, "Obs Max: %0.5f" % (obs["area"].max()), fontsize=10)
    fig.text(0.67, 0.345, "Model Min: %0.5f" % (model["area"].min()), fontsize=10)
    fig.text(0.67, 0.33, "Obs Min: %0.5f" % (obs["area"].min()), fontsize=10)
    fig.text(0.67, 0.285, "Annual Sea Ice Extent", fontsize=10)
    fig.text(0.67, 0.27, "Model Max: %0.5f" % (model["ext"].max()), fontsize=10)
    fig.text(0.67, 0.255, "Obs Max: %0.5f" % (obs["ext"].max()), fontsize=10)
    fig.text(0.67, 0.24, "Model Min: %0.5f" % (model["ext"].min()), fontsize=10)
    fig.text(0.67, 0.225, "Obs Min: %0.5f" % (obs["ext"].min()), fontsize=10)

    # Top header title
    plt.subplots_adjust(top=0.8)
    if region == "nh":
        fig.text(
            0.5,
            0.92,
            "Northern Hemisphere Sea Ice",
            ha="center",
            fontsize=22,
            weight="bold",
        )
    elif region == "sh":
        fig.text(
            0.5,
            0.92,
            "Southern Hemisphere Sea Ice",
            ha="center",
            fontsize=22,
            weight="bold",
        )
    fig.text(
        0.5,
        0.89,
        f"Climatological {month} Sea Ice Concentration  /  "
        + "Annual Cycle of Area and Extent",
        ha="center",
        fontsize=14,
    )
    if label is not None:
        fig.text(0.5, 0.86, label, ha="center", fontsize=14)

    return fig


def run(dictArgs):
    """Function to call read, calc, and plot in sequence"""

    ## parameters
    # interactive = args.interactive
    # outdir = args.outdir
    # pltfmt = args.format
    # infile = args.infile
    # static = args.static
    # month = args.month
    # obsfile = args.obsfile
    # region = args.region
    # label = args.label

    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")

    # --- the main show ---
    ds, dobs, valid_mask = read(dictArgs)

    current_dir = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    model, obs, regridded = calculate(ds, dobs, region=dictArgs["region"])
    os.chdir(current_dir)
    shutil.rmtree(tmpdir)

    fig = plot(
        model,
        obs,
        regridded,
        valid_mask,
        label=dictArgs["label"],
        region=dictArgs["region"],
        month=dictArgs["month"],
    )
    # ---------------------

    filename = f"{dictArgs['outdir']}/seaice.{dictArgs['region']}"
    imgbufs = image_handler([fig], dictArgs, filename=filename)

    return imgbufs


def parse_and_run(cliargs=None):
    """ Function to make compatibile with the superwrapper """
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
