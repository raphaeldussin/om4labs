import xarray as xr
import matplotlib.pyplot as plt

from om4labs.om4common import image_handler
from om4labs.om4common import open_intake_catalog
from om4labs.om4parser import default_diag_parser


def parse(cliargs=None, template=False):
    description = """Plot annual volume transport through the Drake Passage"""

    parser = default_diag_parser(description=description, template=template)

    parser.add_argument("--dataset", type=str, required=False)

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs):
    dset = xr.open_mfdataset(dictArgs["infile"])
    return dset


def ACC_Transport(darray, zdim="z_l", ydim="yh"):
    """Calculate ACC Transport by summing across dimensions

    Parameters
    ----------
    darray : xarray.DataArray
        Data Array containing transport
    zdim : str, optional
        Name of vertical dimension, by default "z_l"
    ydim : str, optional
        Name of latitude dimension, by default "yh"

    Returns
    -------
    xarray.DataArray
        Data Array containing time series of transport
    """
    darray = darray.sum(dim=(zdim, ydim))
    darray = darray * (1.0 / 1035.0) * 1.0e-6
    darray = darray.groupby("time.year").mean(dim="time")

    return darray


def calculate(dset):
    """Performs the ACC calculation by slicing to correct
       y and x regions if using global umo output. Calls
       the ACC_Transport function to compute the transport.

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset containing umo

    Returns
    -------
    xarray.DataArray
        Output DataArray with time series of ACC transport
    """

    # Read in umo from dataset to data array
    darray = dset["umo"]

    # Define a list of acceptable coordinates
    possible_xdims = ["xq", "xh", "xc", "lon", "xq_sub01", "xq_sub02"]
    possible_ydims = ["yh", "yq", "yc", "lat", "yh_sub01", "yh_sub02"]
    possible_zdims = ["z_l", "lev", "level"]

    # set converts our list to a squence of distinct interable elements
    # the intersection method returns a set that contains the items
    # which exist in both set(darray.dims) and set(possible_xdims)
    # and then we turn this back into a list
    xdim = list(set(darray.dims).intersection(set(possible_xdims)))
    ydim = list(set(darray.dims).intersection(set(possible_ydims)))
    zdim = list(set(darray.dims).intersection(set(possible_zdims)))

    # Make sure we have exactly one value per coordinate
    # Immediately trigger error message if condition is false.
    for dim in [xdim, ydim, zdim]:
        assert len(dim) == 1, "Ambiguous coordinates found."

    xdim = xdim[0]
    ydim = ydim[0]
    zdim = zdim[0]

    # If the max latitude is positive (i.e. Northern Hemisphere), assume
    # we have a global array that needs to be subset. Otherwise, just keep
    # the array as-is

    if max(darray[ydim]) > 0.0:
        darray = darray.sel({xdim: -70, ydim: slice(-70, -54)}, method="nearest")
    else:
        darray

    result = ACC_Transport(darray, zdim=zdim, ydim=ydim)

    return result


def plot(dset_out):

    ## Donohue et al. 2016 mean +/- 2 sigma
    x_end = len(dset_out.values)
    x_full = (0, len(dset_out.values))
    max_Donohue = 194.4
    min_Donohue = 151.9
    mean_Donohue = 173.3

    ## Xu et al. 2020 mean +/- 2 sigma
    max_Xu = 161.9
    min_Xu = 152.7
    mean_Xu = 157.3

    ## Biogeochemical Southern Ocean State Estimate (BSOSE) 1/6 degree
    mean_BSOSE = 164

    fig = plt.figure(figsize=(10, 5))
    plt.plot(dset_out.values)
    plt.ylabel("ACC [Sv]", fontweight="bold", fontsize="large")
    plt.xlabel("Model Year", fontweight="bold", fontsize="large")
    plt.grid()

    plt.fill_between(x_full, min_Donohue, max_Donohue, facecolor="grey", alpha=0.5)
    plt.fill_between(x_full, min_Xu, max_Xu, facecolor="yellow", alpha=0.5)
    plt.axhline(y=mean_Donohue, linestyle="--", color="grey")
    plt.axhline(y=mean_Xu, linestyle="--", color="yellow")
    plt.plot(
        x_end,
        mean_Donohue,
        marker="o",
        markersize=10,
        markerfacecolor="grey",
        markeredgecolor="k",
    )
    plt.plot(
        x_end,
        mean_Xu,
        marker="o",
        markersize=10,
        markerfacecolor="yellow",
        markeredgecolor="k",
    )
    plt.plot(
        x_end,
        mean_BSOSE,
        marker="o",
        markersize=10,
        markerfacecolor="cyan",
        markeredgecolor="k",
    )

    fig.text(
        0.92,
        0.67,
        "Donohue et al. 2016 (cDrake)",
        fontsize=12,
        color="black",
        fontweight="bold",
    )
    fig.text(0.92, 0.60, "BSOSE 1/6th", fontsize=12, color="black", fontweight="bold")
    fig.text(
        0.92,
        0.55,
        "Xu et al. 2021 (1/12th HYCOM)",
        fontsize=12,
        color="black",
        fontweight="bold",
    )

    return fig


def run(dictArgs):
    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")

    dset = read(dictArgs)
    dset_out = calculate(dset)
    figs = plot(dset_out)

    filenames = [
        f"{dictArgs['outdir']}/Volume_Transport_Drake",
    ]

    imgbufs = image_handler(figs, dictArgs, filename=filenames)

    return imgbufs


def parse_and_run(cliargs=None):
    """Function to make compatibile with the superwrapper"""
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
