#!/usr/bin/env python3

# Delete anything not used.
import intake
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xesmf as xe
import warnings
from scipy.interpolate import griddata
import copy as copy
from matplotlib.colors import ListedColormap

from om4labs.om4common import image_handler
from om4labs.om4common import open_intake_catalog
from om4labs.om4parser import default_diag_parser

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")

# Do we want a default?  MLD_003 makes sense since it is widely diagnosed.
default_mld = "MLD_003"

# Various domains to plot MLD maps for regional focus.
dims = {
    "global": [-300, 60, -80, 88],
    "NAtl": [-70, 20, 40, 88],
    "EqPac": [120, 300, -25, 25],
    "SO": [-240, 120, -75, -25],
}

# Colorbar limits are specified here for either min/max, the MLD type,
# and the domain.  These are fixed rather than computed from the data
# to ease intercomparison.  An auto override could be added.
cbar_lim = {
    "min": {
        "MLD_003": {
            "global": [0, 60, -20, 20],
            "NAtl": [0, 40, -10, 10],
            "EqPac": [0, 60, -20, 20],
            "SO": [0, 60, -20, 20],
        },
        "MLD_EN1": {
            "global": [0, 60, -20, 20],
            "NAtl": [5, 45, -10, 10],
            "EqPac": [0, 60, -20, 20],
            "SO": [0, 60, -20, 20],
        },
        "MLD_EN2": {
            "global": [0, 150, -20, 20],
            "NAtl": [0, 150, -20, 20],
            "EqPac": [0, 150, -20, 20],
            "SO": [0, 150, -20, 20],
        },
        "MLD_EN3": {
            "global": [0, 1000, -200, 200],
            "NAtl": [0, 1000, -200, 200],
            "EqPac": [0, 1000, -200, 200],
            "SO": [0, 1000, -200, 200],
        },
    },
    "max": {
        "MLD_003": {
            "global": [0, 500, -100, 100],
            "NAtl": [0, 500, -100, 100],
            "EqPac": [0, 500, -100, 100],
            "SO": [0, 500, -100, 100],
        },
        "MLD_EN1": {
            "global": [0, 500, -100, 100],
            "NAtl": [0, 500, -100, 100],
            "EqPac": [0, 500, -100, 100],
            "SO": [0, 500, -100, 100],
        },
        "MLD_EN2": {
            "global": [0, 500, -100, 100],
            "NAtl": [0, 500, -100, 100],
            "EqPac": [0, 500, -100, 100],
            "SO": [0, 500, -100, 100],
        },
        "MLD_EN3": {
            "global": [0, 1000, -200, 200],
            "NAtl": [0, 1000, -200, 200],
            "EqPac": [0, 1000, -200, 200],
            "SO": [0, 1000, -200, 200],
        },
    },
}


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
        "--mldvar",
        type=str,
        default=default_mld,
        help="MLD variable to analyze. Default is " + default_mld + ".",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="max",
        help="Maximum monthly MLDs (Winter) or minimum MLDs (Summer).  Default is max.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="global",
        help="Grid: 'global', 'NAtl' (North Atlantic), 'EqPac' (Equatorial Pacific), or 'SO' (Southern Ocean).  Default is global.",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs):
    """Read required fields to plot MLD

    Parameters
    ----------
    dictArgs : dict
        Dictionary containing argparse options

    Returns
    -------
    ds_model : xarray.DataSet
        DataSet containing all the necessary model data.
    ds_obs : xarray.DataSet
        DataSet containing all the necessary obs data.
    ds_static : xarray.DataSet
        DataSet containing the static file for the model data.
    """
    ds_input = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)
    ds_static = xr.open_mfdataset(dictArgs["static"])

    # If obsfile is set, load it.  If not, choose a default dataset.
    if dictArgs["obsfile"] is not None:
        ds_obs = xr.open_dataset(dictArgs["obsfile"])
    else:
        mldvar = dictArgs["mldvar"]
        if mldvar == "MLD_EN1":
            cat = open_intake_catalog(dictArgs["platform"], "obs")
            ds_obs = cat["Argo_MLD_EN1"].to_dask()
        elif mldvar == "MLD_EN2":
            cat = open_intake_catalog(dictArgs["platform"], "obs")
            ds_obs = cat["Argo_MLD_EN2"].to_dask()
        elif mldvar == "MLD_EN3":
            cat = open_intake_catalog(dictArgs["platform"], "obs")
            ds_obs = cat["Argo_MLD_EN3"].to_dask()
        elif mldvar == "MLD_003":
            cat = open_intake_catalog(dictArgs["platform"], "obs")
            ds_obs = cat["Argo_MLD_003"].to_dask()

    ds_model = ds_input[mldvar].groupby("time.month").mean("time")

    return (ds_model, ds_obs, ds_static)


def calculate(ds_model, ds_obs, ds_static, dictArgs):
    """Main calculation function

    Parameters
    ----------
    ds_model : xarray.Dataset
        Input dataset including model data
    ds_obs : xarray.Dataset
        Input dataset including obs data
    ds_static : xarray.Dataset
        Input static dataset for model
    dictArgs : dictionary
        Input dictionary storing options

    Returns
    -------
    ds_plot : xarray.DataArray
        Ouput dataset including all data needed to generate plots
    """

    ds_plot = xr.Dataset()

    # Using the obs to build the plotting grid

    # The plot dimensions are set by the grid choice.
    LonMin = dims[dictArgs["grid"]][0]
    LonMax = dims[dictArgs["grid"]][1]
    LatMin = dims[dictArgs["grid"]][2]
    LatMax = dims[dictArgs["grid"]][3]

    # Extract copy of obs domain
    obs_lat = ds_obs["Lat"].values
    obs_lon = ds_obs["Lon"].values
    # Adjust obs_lon to fit in specified dimensions (this assumes they are within 360, should be...)
    obs_lon[obs_lon < LonMin] += 360
    obs_lon[obs_lon > LonMax] -= 360
    xi = np.argsort(obs_lon)
    obs_lon_sort = obs_lon[xi]
    # Mask points within domain
    lonlims = np.where((obs_lon_sort > LonMin) & (obs_lon_sort < LonMax))[0]
    latlims = np.where((obs_lat > LatMin) & (obs_lat < LatMax))[0]
    # Extract points within domain and set as common grid
    cmn_lat = obs_lat[latlims]
    cmn_lon = obs_lon_sort[lonlims]
    # Fill plot dataset w/ 1d data
    ds_plot["lat"] = cmn_lat
    ds_plot["lon"] = cmn_lon
    # Create 2d maps for griddata routine
    # -First for the obs data
    obs_lat, obs_lon = np.meshgrid(obs_lat, obs_lon)
    # -Second for the common grid, based on extracted obs grid
    plon, plat = np.meshgrid(cmn_lon, cmn_lat)

    method = dictArgs["method"]

    if method == "min":
        model_data = ds_model.min(dim="month")
        obs_data = ds_obs.MLD_mean.min(dim="Month", skipna=False)
    elif method == "max":
        model_data = ds_model.max(dim="month")
        obs_data = ds_obs.MLD_mean.max(dim="Month", skipna=False)

    model = xr.Dataset()
    model["yh"] = ds_model.yh.values
    model["xh"] = ds_model.xh.values
    model["MLD"] = (("yh", "xh"), model_data.values)
    model = model.assign_coords(lon=(("yh", "xh"), ds_static["geolon"].values))
    model = model.assign_coords(lat=(("yh", "xh"), ds_static["geolat"].values))

    regridder_mod = xe.Regridder(model, ds_plot, "bilinear", periodic=True)

    ds_plot["model"] = (("lat", "lon"), regridder_mod(model).MLD.values)
    ds_plot["obs"] = (
        ("lat", "lon"),
        (
            ((obs_data.values.T)[:, xi])[
                latlims[0] : latlims[-1] + 1, lonlims[0] : lonlims[-1] + 1
            ]
        ),
    )

    # Want to also compute the metrics here (bias, RMS, r2)
    # First we need to max out any NaN data.
    diff = ds_plot.model.values - ds_plot.obs.values
    Msk = (
        (np.isfinite(ds_plot.model.values))
        & (np.isfinite(ds_plot.obs.values))
        & (plon > LonMin)
        & (plon < LonMax)
        & (plat > LatMin)
        & (plat < LatMax)
    )

    # Note that we are area weighting our global metrics by assuming a spherical Earth.  This is a decent approximation and better than not area weighting the metrics.
    ds_plot["bias"] = np.nansum(
        (diff[Msk] * np.cos(plat[Msk] * np.pi / 180.0)).ravel()
    ) / np.nansum((np.cos(plat[Msk] * np.pi / 180.0)).ravel())
    ds_plot["RMS"] = np.sqrt(
        np.nansum((diff[Msk] ** 2 * np.cos(plat[Msk] * np.pi / 180.0)).ravel())
        / np.nansum((np.cos(plat[Msk] * np.pi / 180.0)).ravel())
    )
    ds_plot["r2"] = (
        np.corrcoef(ds_plot.model.values[Msk].ravel(), ds_plot.obs.values[Msk].ravel())[
            1, 0
        ]
        ** 2
    )

    return ds_plot


def plot(ds_plot, dictArgs):
    """Plotting wrapper

    Parameters
    ----------
    ds_plot : xarray.DataArray
        Input data array

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle
    """

    # Pull some options out of dictionary for easier code reading
    method = dictArgs["method"]
    mldvar = dictArgs["mldvar"]
    grid = dictArgs["grid"]

    cmap = copy.copy(plt.cm.viridis_r)
    cmap2 = copy.copy(plt.cm.PuOr)
    cmap.set_bad("gray")
    cmap2.set_bad("gray")

    # The level limits are set above in this file.
    levels = np.linspace(
        cbar_lim[method][mldvar][grid][0], cbar_lim[method][mldvar][grid][1], 21
    )
    levels2 = np.linspace(
        cbar_lim[method][mldvar][grid][2], cbar_lim[method][mldvar][grid][3], 21
    )

    fig = plt.figure(figsize=(12, 6))
    # Extract plot data
    lon = ds_plot["lon"].values
    lat = ds_plot["lat"].values
    mod = ds_plot["model"].values
    obs = ds_plot["obs"].values
    # Compute difference field
    dif = mod - obs
    lon, lat = np.meshgrid(lon, lat)

    # Consider alternate map projections?
    # ax=fig.add_axes([0.1,0.52,0.425,0.32],projection=ccrs.Robinson(central_longitude=dictArgs["central"]),facecolor='gray',)
    ax = fig.add_axes([0.1, 0.52, 0.4, 0.32])
    cb0 = ax.pcolormesh(
        lon,
        lat,
        mod,
        shading="auto",
        # transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False),
    )
    # Add hatching to cross out areas in model that are not in obs.  This should make it easier to compare panels.
    notobs = np.copy(obs) * np.NaN
    notobs[(np.isnan(obs)) & (np.isfinite(mod))] = 1
    ax.pcolor(
        lon, lat, notobs, hatch="+", cmap=ListedColormap(["none"]), shading="auto"
    )
    ax.set_title("Model")

    # ax=fig.add_axes([0.5,0.52,0.425,0.32],projection=ccrs.Robinson(central_longitude=dictArgs["central"]),facecolor='gray')
    ax = fig.add_axes([0.55, 0.52, 0.4, 0.32])
    cb1 = ax.pcolormesh(
        lon,
        lat,
        obs,
        shading="auto",
        # transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False),
    )
    ax.set_title("Argo")

    # ax=fig.add_axes([0.1,0.1,0.425,0.32],projection=ccrs.Robinson(central_longitude=dictArgs["central"]),facecolor='gray')
    ax = fig.add_axes([0.1, 0.1, 0.4, 0.32])
    cb2 = ax.pcolormesh(
        lon,
        lat,
        dif,
        shading="auto",
        # transform=ccrs.PlateCarree(),
        cmap=cmap2,
        norm=mpl.colors.BoundaryNorm(levels2, ncolors=cmap2.N, clip=False),
    )
    ax.set(title="Model - Argo")

    cax1 = fig.add_axes([0.55, 0.35, 0.4, 0.02])
    cbar1 = plt.colorbar(cb1, cax=cax1, orientation="horizontal")
    if dictArgs["mldvar"] is not None:
        cbar1.set_label(dictArgs["mldvar"] + " [m]")
    else:
        cbar1.set_label(default_mld + " [m]")
    cax2 = fig.add_axes([0.55, 0.2, 0.4, 0.02])
    cbar2 = plt.colorbar(cb2, cax=cax2, orientation="horizontal")
    cbar2.set_label(dictArgs["mldvar"] + " difference [m]")

    ax = fig.add_axes([0.1, 0.9, 0.8, 0.1])
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    FS = 12
    ax.text(
        0.25,
        0.2,
        "Bias={:4.3f}".format(ds_plot["bias"].values),
        fontsize=FS,
        horizontalalignment="center",
    )
    ax.text(
        0.5,
        0.2,
        "RMS={:4.3f}".format(ds_plot["RMS"].values),
        fontsize=FS,
        horizontalalignment="center",
    )
    ax.text(
        0.75,
        0.2,
        "$r^2$={:4.3f}".format(ds_plot["r2"].values),
        fontsize=FS,
        horizontalalignment="center",
    )
    FS = 16
    ax.text(
        0.5,
        0.8,
        dictArgs["label"] + ": Monthly " + dictArgs["method"],
        fontsize=FS,
        horizontalalignment="center",
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
    ds_model, ds_obs, ds_static = read(dictArgs)

    # calculate
    ds_plot = calculate(ds_model, ds_obs, ds_static, dictArgs)

    # make the plots
    fig = plot(ds_plot, dictArgs)

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
