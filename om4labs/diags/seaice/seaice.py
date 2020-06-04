#!/usr/bin/env python3

"""
om4labs: model-simulated sea ice vs. NSIDC obs
"""

__all__ = ["arguments", "read", "calculate", "plot", "run", "parse_and_run"]

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature
import time
import warnings

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")

try:
    from . import averagers
    from . import regrid
except:
    import averagers
    import regrid
import io
import numpy as np
import palettable
import xarray as xr


def arguments(cliargs=None):
    description = """Plot sea ice vs. NSIDC"""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "infile",
        metavar="INFILE",
        type=str,
        nargs="+",
        help="Path to input NetCDF file(s)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./",
        help="Output directory. Default is current directory",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="png",
        help="Output format for plots. Default is png",
    )

    parser.add_argument(
        "-m",
        "--month",
        type=str,
        default="March",
        help="Month to analyze. Deafult is March",
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode displays plot to screen. Default is False",
    )

    parser.add_argument(
        "-s", "--static", type=str, default=None, help="Path to static file"
    )

    parser.add_argument(
        "-r",
        "--region",
        type=str,
        default="nh",
        help="Region for analysis. Default is nh. Options are nh and sh",
    )

    parser.add_argument(
        "-O",
        "--obsfile",
        type=str,
        default=None,
        help="Path to file containing observations",
    )

    parser.add_argument(
        "-l", "--label", type=str, default="", help="String label to annotate the plot"
    )

    return parser.parse_args(cliargs)


def read(infile, obsfile, static):
    """Function to read in the data. Returns xarray datasets"""

    # Open ice model output and the static file
    ds = xr.open_mfdataset(infile, combine="by_coords")
    if "siconc" in ds.variables:
        ds["CN"] = ds["siconc"]
        if ds["CN"].max() > 1.0:
            ds["CN"] = ds["CN"] / 100.0
    else:
        ds["CN"] = ds["CN"].sum(dim="ct")
    dstatic = xr.open_dataset(static)
    ds["CELL_AREA"] = dstatic["CELL_AREA"]
    ds["GEOLON"] = dstatic["GEOLON"]
    ds["GEOLAT"] = dstatic["GEOLAT"]
    ds["AREA"] = dstatic["CELL_AREA"] * 4.0 * np.pi * (6.378e6 ** 2)

    # Get Valid Mask
    valid_mask = np.where(ds["CELL_AREA"] == 0, True, False)

    # Open observed SIC on 25-km EASE grid (coords already named lat and lon)
    dobs = xr.open_dataset(obsfile)

    # Close the static file (no longer used)
    dstatic.close()

    return ds, dobs, valid_mask


def calculate(ds, dobs, region="nh"):
    """ Function to calculate sea ice parameters """

    # Container dictionaries to hold results
    model = {}
    obs = {}

    # Create annual cycle climatology
    model["ac"] = averagers.annual_cycle(ds, "CN")
    obs["ac"] = averagers.annual_cycle(dobs, "sic")

    # Calculate area and extent
    if region == "nh":
        model["area"] = np.where(ds["GEOLAT"] > 0.0, model["ac"] * ds.AREA, 0.0)
        model["ext"] = np.where(
            (model["ac"] > 0.15) & (ds["GEOLAT"] > 0.0), ds.AREA, 0.0
        )
    elif region == "sh":
        model["area"] = np.where(ds["GEOLAT"] < 0.0, model["ac"] * ds.AREA, 0.0)
        model["ext"] = np.where(
            (model["ac"] > 0.15) & (ds["GEOLAT"] < 0.0), ds.AREA, 0.0
        )
    else:
        raise ValueError(f"Unknown region {region}. Option are nh or sh")

    model["area"] = model["area"].sum(axis=(-2, -1)) * 1.0e-12
    model["ext"] = model["ext"].sum(axis=(-2, -1)) * 1.0e-12

    obs["area"] = obs["ac"] * dobs.areacello
    obs["area"] = obs["area"].sum(axis=(-2, -1)) * 1.0e-12

    obs["ext"] = np.where(obs["ac"] > 0.15, dobs.areacello, 0.0)
    obs["ext"] = obs["ext"].sum(axis=(-2, -1)) * 1.0e-12

    # Add back in the 2D coordinates
    model["ac"]["GEOLON"] = ds["GEOLON"]
    model["ac"]["GEOLAT"] = ds["GEOLAT"]
    model["ac"] = model["ac"].rename({"GEOLON": "lon", "GEOLAT": "lat"})
    obs["ac"]["lon"] = dobs["lon"]
    obs["ac"]["lat"] = dobs["lat"]

    # Regrid the observations to the model grid (for plotting)
    obs["ac_r"] = regrid.curv_to_curv(obs["ac"], model["ac"], reuse_weights=False)

    # Get tuple of start year and end years for model and observations
    model["time"] = (
        int(ds["time"].isel({"time": 0}).dt.strftime("%Y")),
        int(ds["time"].isel({"time": -1}).dt.strftime("%Y")),
    )

    obs["time"] = (
        int(dobs["time"].isel({"time": 0}).dt.strftime("%Y")),
        int(dobs["time"].isel({"time": -1}).dt.strftime("%Y")),
    )

    return model, obs


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
    cmap = plt.get_cmap(cmap)
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


def plot(model, obs, valid_mask, label=None, region="nh", month="March"):
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
    x = np.array(model["ac"].lon.to_masked_array())
    y = np.array(model["ac"].lat.to_masked_array())

    # Top left panel - model map of sea ice
    ax = plt.subplot(2, 3, 1, projection=proj)
    plotdata = (model["ac"][month_index] * 100.0).to_masked_array()
    plotdata = np.ma.masked_where(valid_mask, plotdata)
    cb1 = _plot_map_panel(ax, x, y, plotdata, extent=extent)
    ax.set_title(f"Model - Years {model['time'][0]} to {model['time'][1]}")
    fig.colorbar(
        cb1, orientation="horizontal", fraction=0.03, pad=0.05, aspect=60, ax=ax
    )

    # Top middle panel - observed map of sea ice
    ax = plt.subplot(2, 3, 2, projection=proj)
    plotdata = (obs["ac_r"][month_index] * 100.0).to_masked_array()
    plotdata = np.ma.masked_where(valid_mask, plotdata)
    cb2 = _plot_map_panel(ax, x, y, plotdata, extent=extent)
    ax.set_title(f"NSIDC - Years {obs['time'][0]} to {obs['time'][1]}")
    fig.colorbar(
        cb2, orientation="horizontal", fraction=0.03, pad=0.05, aspect=60, ax=ax
    )

    # Top right panel - model minus observed difference
    ax = plt.subplot(2, 3, 3, projection=proj)
    _mod = np.where(
        np.isnan(model["ac"][month_index].data), 0.0, model["ac"][month_index].data
    )
    _obs = np.where(
        np.isnan(obs["ac_r"][month_index].data), 0.0, obs["ac_r"][month_index].data
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


def run(args):
    """Function to call read, calc, and plot in sequence"""

    # parameters
    interactive = args.interactive
    outdir = args.outdir
    pltfmt = args.format
    infile = args.infile
    static = args.static
    month = args.month
    obsfile = args.obsfile
    region = args.region
    label = args.label

    # set visual backend
    if interactive is False:
        plt.switch_backend("Agg")
    else:
        plt.switch_backend("TkAgg")

    print(f"Matplotlib is using the {mpl.get_backend()} back-end.")

    # --- the main show ---
    ds, dobs, valid_mask = read(infile, obsfile, static)
    model, obs = calculate(ds, dobs, region=region)
    fig = plot(model, obs, valid_mask, label=label, region=region, month=month)
    # ---------------------

    # do something with the figure
    if interactive is True:
        plt.show(fig)
    else:
        imgbuf = io.BytesIO()
        fig.savefig(imgbuf, format=pltfmt, dpi=150, bbox_inches="tight")
        with open(f"icefig.{pltfmt}", "wb") as f:
            f.write(imgbuf.getbuffer())


def parse_and_run(cliargs=None):
    """ Function to make compatibile with the superwrapper """
    args = arguments(cliargs)
    run(args)


if __name__ == "__main__":
    parse_and_run()
