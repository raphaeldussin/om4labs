import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import xcompare
import xarray as xr
from xgcm import Grid

from xcompare import plot_three_panel
from om4labs.om4common import image_handler
from om4labs.om4common import open_intake_catalog
from om4labs.om4common import date_range
from om4labs.om4parser import default_diag_parser


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

    example usage:
    om4labs stress_curl -s fname_static.nc --dataset OM4_windstress --period 1999-2018 fname_model_taux.nc fname_model_tauy.nc
    """

    description = " "

    exclude = ["basin", "hgrid", "topog", "gridspec", "config"]

    parser = default_diag_parser(
        description=description, template=template, exclude=exclude
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="OM4_windstress",
        help="Name of the observational dataset, \
              as provided in intake catalog",
    )

    parser.add_argument(
        "--period",
        type=str,
        required=False,
        default="1959-1978",
        help="Time period for OMIP2 dataset, available are 1959-1978 and 1999-2018",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs):
    ds_model = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)
    dates = date_range(ds_model)

    if dictArgs["obsfile"] is not None:
        # priority to user-provided obs file
        dsobs = xr.open_mfdataset(
            dictArgs["obsfile"], combine="by_coords", decode_times=False
        )
    else:
        # use dataset from catalog, either from command line or default
        cat = open_intake_catalog(dictArgs["platform"], "OMIP2")
        obsdataset = f"{dictArgs['dataset']}_{dictArgs['period']}"
        ds_ref_tau = cat[obsdataset].to_dask()

    ds_static = xr.open_mfdataset(dictArgs["static"])

    # replace the nominal xq and yq by indices so that Xarray does not get confused.
    # Confusion arises since there are inconsistencies between static file grid and
    # model data grid for the last value of yq. We never need xq and yq for actual
    # calculations, so filling these arrays with any value is not going to change
    # any results. But Xarray needs them to be consistent between the two files when
    # doing the curl operation on the stress.
    ds_model["xq"] = xr.DataArray(np.arange(len(ds_model["xq"])), dims=["xq"])
    ds_model["yq"] = xr.DataArray(np.arange(len(ds_model["yq"])), dims=["yq"])
    ds_ref_tau["xq"] = xr.DataArray(np.arange(len(ds_ref_tau["xq"])), dims=["xq"])
    ds_ref_tau["yq"] = xr.DataArray(np.arange(len(ds_ref_tau["yq"])), dims=["yq"])
    ds_static["xq"] = xr.DataArray(np.arange(len(ds_static["xq"])), dims=["xq"])
    ds_static["yq"] = xr.DataArray(np.arange(len(ds_static["yq"])), dims=["yq"])

    ds_model.attrs = {"date_range": dates}

    return ds_model, ds_ref_tau, ds_static


def calc_curl_stress(
    ds,
    ds_static,
    varx="tauuo",
    vary="tauvo",
    areacello_bu="areacello_bu",
    xdim="lon",
    ydim="lat",
    rho0=1035.0,
    maskname="wet_c",
    lonname="geolon_c",
    latname="geolat_c",
):
    """Calculate curl of stress acting on surface of the ocean.

    Parameters
        ----------
    ds        : xarray.Dataset dataset with tauuo and tauvo
    ds_static : xarray.Dataset with grid values
    varname   : str, optional
        Name of the tauuo and tauvo variables, by default "tauuo" and "tauvo"
    area : str, optional
        Name of the area variable, by default "areacello_bu"
    xdim : str, optional
        Name of the longitude coordinate, by default "lon"
    ydim : str, optional
        Name of the latitude coordinate, by default "lat"
    rho0: float, optional
        Reference density of seawater, by default 1035.0 kg/m3
    maskname: str, optional
        Name of land/sea mask, by default wet_c
    lonname: str, optional
        Name of longitude variable, by default geolon_c
    latname: str, optional
        Name of latitude variable, by default geolat_c
    Returns
    -------
        xarray.DataArray stress_curl
        curl of surface ocean stress
    """

    area = ds_static[areacello_bu]
    taux = ds[varx].mean(dim="time")
    tauy = ds[vary].mean(dim="time")

    # fill nan with 0.0 since want 0.0 values over land for the curl operation
    taux = taux.fillna(0.0)
    tauy = tauy.fillna(0.0)

    grid = Grid(
        ds_static,
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
        },
        periodic=["X"],
    )

    stress_curl = -grid.diff(taux * ds_static.dxCu, "Y", boundary="fill") + grid.diff(
        tauy * ds_static.dyCv, "X", boundary="fill"
    )

    stress_curl = stress_curl / (area * rho0)
    stress_curl = stress_curl.where(ds_static[maskname] == 1)
    stress_curl = stress_curl.assign_coords(
        {lonname: ds_static[lonname], latname: ds_static[latname]}
    )

    return stress_curl


def calculate(ds_model, ds_ref_tau, ds_static, dictArgs):

    curl_model = xr.Dataset(
        {
            "stress_curl": calc_curl_stress(ds_model, ds_static),
            "areacello_bu": ds_static["areacello_bu"],
        }
    )

    curl_ref = xr.Dataset(
        {
            "stress_curl": calc_curl_stress(ds_ref_tau, ds_static),
            "areacello_bu": ds_static["areacello_bu"],
        }
    )

    results = xcompare.compare_datasets(curl_model, curl_ref)

    return results


def plot(dictArgs, results):

    fig = xcompare.plot_three_panel(
        results,
        "stress_curl",
        cmap=cmocean.cm.delta,
        projection=ccrs.Robinson(),
        coastlines=False,
        vmin=-3e-10,
        vmax=3e-10,
        diffvmin=-3e-10,
        diffvmax=3e-10,
        labels=[dictArgs["label"], "OMIP reference"],
        lon_range=(-180, 180),
        lat_range=(-90, 90),
    )
    return fig


def plot_old(
    field,
    vmin=-3e-10,
    vmax=3e-10,
    lat_lon_ext=[-180, 180, -85.0, 90.0],
    lon="geolon_c",
    lat="geolat_c",
    cmap=cmocean.cm.delta,
    title="Curl of stress (N/m**2) acting on ocean surface",
):

    # convert xarray to ordinary numpy arrays
    if isinstance(field, xr.DataArray):
        geolon = field[lon].values
        geolat = field[lat].values
        field = field.values

    fig = plt.figure(figsize=[22, 8])
    ax = fig.add_subplot(projection=ccrs.Robinson(), facecolor="grey")
    cb = ax.pcolormesh(
        geolon,
        geolat,
        field,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    # add separate colorbar
    cb = plt.colorbar(cb, ax=ax, format="%.1e", extend="both", shrink=0.6)
    cb.ax.tick_params(labelsize=12)

    # add gridlines and extent of lat/lon
    ax.gridlines(color="black", alpha=0.5, linestyle="--")
    ax.set_extent(lat_lon_ext, crs=ccrs.PlateCarree())
    _ = plt.title(title, fontsize=14)

    return fig


def run(dictArgs):
    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")

    ds_model, ds_ref_tau, ds_static = read(dictArgs)

    results = calculate(ds_model, ds_ref_tau, ds_static, dictArgs)
    figs = plot(dictArgs, results)
    figs = [figs] if not isinstance(figs, list) else figs
    assert isinstance(figs, list), "Figures must be inside a list object"

    filenames = [
        f"{dictArgs['outdir']}/surface_stress_curl",
    ]

    imgbufs = image_handler(figs, dictArgs, filename=filenames)

    return imgbufs


def parse_and_run(cliargs=None):
    """Parses command line and runs diagnostic

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
