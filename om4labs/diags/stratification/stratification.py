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
from om4labs import m6plot
import gsw as gsw

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
from om4labs.om4common import standardize_longitude


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

    parser.add_argument(
        "--argo_temp_file",
        type=str,
        required=False,
        default=None,
        help="Name of the Argo in-situ temperature file",
    )

    parser.add_argument(
        "--argo_psal_file",
        type=str,
        required=False,
        default=None,
        help="Name of the Argo practical salinity file",
    )

    parser.add_argument(
        "--model_xcoord",
        type=str,
        required=False,
        default="lon",
        help="Name of x-coordinate in the model data",
    )

    parser.add_argument(
        "--model_ycoord",
        type=str,
        required=False,
        default="lat",
        help="Name of y-coordinate in the model data",
    )

    parser.add_argument(
        "--model_zcoord",
        type=str,
        required=False,
        default="z_l",
        help="Name of z-coordinate in the model data",
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

    tempvar = "thetao"
    saltvar = "so"

    model_xcoord = dictArgs["model_xcoord"]
    model_ycoord = dictArgs["model_ycoord"]
    model_zcoord = dictArgs["model_zcoord"]

    model = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)
    model = xr.Dataset(
        {
            "temp": model[tempvar].mean(dim="time"),
            "salt": model[saltvar].mean(dim="time"),
        }
    )

    # clean up and standardize
    model = model.squeeze()
    model = model.reset_coords(drop=True)
    model = model.rename({model_xcoord: "lon", model_ycoord: "lat"})

    # ----- read the Argo data
    obs_xcoord = "LONGITUDE"
    obs_ycoord = "LATITUDE"
    obs_zcoord = "PRESSURE"

    if (dictArgs["argo_temp_file"] is not None) or (
        dictArgs["argo_psal_file"] is not None
    ):

        assert (
            dictArgs["argo_temp_file"] is not None
            and dictArgs["argo_psal_file"] is not None
        ), "Both Ago temp and psal files must be specified for consistency"

        argo_dset = xr.open_mfdataset(
            [dictArgs["argo_temp_file"], dictArgs["argo_psal_file"]],
            combine="by_coords",
            decode_times=False,
        )
    else:
        # use dataset from catalog, either from command line or default
        cat = open_intake_catalog(dictArgs["platform"], "obs")
        argo_dset = cat["Argo_Climatology"].to_dask()

    # standardize longitude to go from 0 to 360.
    argo_dset = standardize_longitude(argo_dset, obs_xcoord)

    # subset varaibles
    argo_dset = xr.Dataset(
        {
            "temp": argo_dset["ARGO_TEMPERATURE_MEAN"],
            "salt": argo_dset["ARGO_SALINITY_MEAN"],
            "pres": argo_dset["PRESSURE"],
        }
    )

    # clean up and standardize
    argo_dset = argo_dset.squeeze()
    argo_dset = argo_dset.reset_coords(drop=True)
    argo_dset = argo_dset.rename({obs_xcoord: "lon", obs_ycoord: "lat"})

    # ----- read the WOA data

    if dictArgs["obsfile"] is not None:
        # priority to user-provided obs file
        dsobs = xr.open_mfdataset(
            dictArgs["obsfile"], combine="by_coords", decode_times=False
        )
    else:
        # use dataset from catalog, either from command line or default
        cat = open_intake_catalog(dictArgs["platform"], "obs")
        dsobs = cat[dictArgs["dataset"]].to_dask()

    woa = xr.Dataset(
        {
            "temp": dsobs["ptemp"],
            "salt": dsobs["salinity"],
        }
    )
    woa = woa.squeeze()
    woa = woa.reset_coords(drop=True)

    return model, woa, argo_dset


def calculate(
    model,
    woa,
    argo_dset,
    zcoord="z_l",
    model_xcoord="lon",
    model_ycoord="lat",
    argo_xcoord="lon",
    argo_ycoord="lat",
    argo_zcoord="PRESSURE",
    depth_range=(25, 1000),
):
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

    # -- Model Data

    pottemp = model["temp"]

    dtdz = (-1.0 * pottemp).differentiate(zcoord).sel({zcoord: slice(*depth_range)})
    dtdz.load()
    mask = xr.where(dtdz.isel({zcoord: 0}).isnull(), np.nan, 1.0)
    dtdz = dtdz.fillna(0.0)
    dtdz_max = dtdz.max(zcoord)
    dtdz_max_depth = xr.broadcast(dtdz.z_l, dtdz)[0][dtdz.argmax(zcoord)]
    dtdz_model_max = xr.Dataset(
        {"dtdz_max": dtdz_max * mask, "dtdz_max_depth": dtdz_max_depth * mask}
    )

    # define regions for model data
    # _xval = pottemp[model_xcoord].sel({model_xcoord: -140.0}, method="nearest")
    _xval = pottemp[model_xcoord].sel({model_xcoord: 220.0}, method="nearest")
    pacific = {model_xcoord: _xval, model_ycoord: slice(-70.0, 60.0)}
    # _xval = pottemp[model_xcoord].sel({model_xcoord: -30.0}, method="nearest")
    _xval = pottemp[model_xcoord].sel({model_xcoord: 330.0}, method="nearest")
    atlantic = {model_xcoord: _xval, model_ycoord: slice(-70.0, 80.0)}

    # extract sections for comparison with Argo
    basin = xr.DataArray(np.array(["atlantic", "pacific"]), dims=("basin"))
    model = xr.concat([pottemp.sel(atlantic), pottemp.sel(pacific)], dim="basin")
    model = model.assign_coords({"basin": basin})

    dtdz_model = (-1.0 * model).differentiate(zcoord).squeeze()

    # -- WOA

    dtdz = (-1.0 * woa.temp).differentiate(zcoord).sel({zcoord: slice(*depth_range)})
    dtdz.load()
    mask = xr.where(dtdz.isel({zcoord: 0}).isnull(), np.nan, 1.0)
    dtdz = dtdz.fillna(0.0)
    dtdz_max = dtdz.max(zcoord)
    dtdz_max_depth = xr.broadcast(dtdz.z_l, dtdz)[0][dtdz.argmax(zcoord)]
    dtdz_woa_max = xr.Dataset(
        {"dtdz_max": dtdz_max * mask, "dtdz_max_depth": dtdz_max_depth * mask}
    )

    # -- ARGO

    _xval = argo_dset[argo_xcoord].sel({argo_xcoord: 220.0}, method="nearest")
    pacific = {argo_xcoord: _xval, argo_ycoord: slice(-70.0, 60.0)}

    _xval = argo_dset[argo_xcoord].sel({argo_xcoord: 330.0}, method="nearest")
    atlantic = {argo_xcoord: _xval, argo_ycoord: slice(-70.0, 80.0)}

    argo_dset = xr.concat(
        [argo_dset.sel(atlantic), argo_dset.sel(pacific)], dim="basin"
    )
    argo_dset = argo_dset.assign_coords({"basin": basin})

    argo_dset = xr.Dataset(
        {
            "thetao": xr.DataArray(
                gsw.pt_from_t(
                    argo_dset["salt"], argo_dset["temp"], argo_dset["pres"], 0.0
                ),
                name="thetao",
                attrs={"long_name": "Seawater Potential Temperature", "units": "deg C"},
            )
        }
    )

    dtdz_argo = (-1.0 * argo_dset.thetao).differentiate(argo_zcoord).squeeze()

    max_comparison_results = xcompare.compare_datasets(dtdz_model_max, dtdz_woa_max)

    return max_comparison_results, dtdz_model, dtdz_argo


def _plot_basin(
    dtdz_model, dtdz_argo, basin="atlantic", model_zcoord="z_l", argo_zcoord="PRESSURE"
):

    cmap = plt.cm.RdYlBu_r
    levels = [
        1.0e-7,
        5.0e-7,
        1.0e-6,
        5.0e-6,
        1.0e-5,
        5.0e-5,
        1.0e-4,
        5.0e-4,
        1.0e-3,
        5.0e-3,
    ] + list(np.linspace(0.01, 0.2, 10))
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    fig = plt.figure(figsize=(18, 5))
    ax1 = plt.subplot(1, 2, 1, facecolor="gray")
    plotarr = dtdz_model.sel(basin=basin)
    cb = ax1.pcolormesh(
        plotarr.lat,
        plotarr[model_zcoord],
        plotarr.values,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    plt.plot(plotarr.lat, (plotarr.lat * 0.0) + 2000, "k--")
    plt.colorbar(cb)
    ax1.set_yscale("splitscale", zval=[5500, 2000, 0])
    ax1.set_xlim(-70, 70)

    ax2 = plt.subplot(1, 2, 2, facecolor="gray")
    plotarr = dtdz_argo.sel(basin=basin)
    cb = ax2.pcolormesh(
        plotarr.lat,
        plotarr[argo_zcoord],
        plotarr.values,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    plt.plot(plotarr.lat, (plotarr.lat * 0.0) + 2000, "k--")
    plt.colorbar(cb)
    ax2.set_yscale("splitscale", zval=[5500, 2000, 0])
    ax2.set_xlim(-70, 70)

    ax1.text(
        0.01,
        1.02,
        f"a. {basin.capitalize()} Sector dT/dz",
        ha="left",
        transform=ax1.transAxes,
    )
    ax1.text(0.98, 1.02, f"Model", ha="right", transform=ax1.transAxes)
    ax2.text(
        0.01,
        1.02,
        f"a. {basin.capitalize()} Sector dT/dz",
        ha="left",
        transform=ax2.transAxes,
    )
    ax2.text(0.98, 1.02, f"Argo", ha="right", transform=ax2.transAxes)

    return fig


def plot(max_comparison_results, dtdz_model, dtdz_argo):
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

    figs = [
        xcompare.plot_three_panel(
            max_comparison_results, "dtdz_max", diffvmin=-0.02, diffvmax=0.02
        ),
        xcompare.plot_three_panel(
            max_comparison_results, "dtdz_max_depth", diffvmin=-150, diffvmax=150
        ),
    ]

    figs.append(_plot_basin(dtdz_model, dtdz_argo, "atlantic"))
    figs.append(_plot_basin(dtdz_model, dtdz_argo, "pacific"))

    return figs


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
    model, woa, argo_dset = read(dictArgs)

    # calculate
    max_comparison_results, dtdz_model, dtdz_argo = calculate(model, woa, argo_dset)

    # make the plots
    figs = plot(max_comparison_results, dtdz_model, dtdz_argo)

    filename = [
        f"{dictArgs['outdir']}/thermocline_strength",
        f"{dictArgs['outdir']}/thermocline_depth",
        f"{dictArgs['outdir']}/argo_comparison_atlantic",
        f"{dictArgs['outdir']}/argo_comparison_pacific",
    ]
    imgbufs = image_handler(figs, dictArgs, filename=filename)

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
