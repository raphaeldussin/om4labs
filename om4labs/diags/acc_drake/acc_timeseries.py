def parse():
    return


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
    darray = darray * (1. / 1035.) * 1.0e-6
    darray = darray.groupby("time.year").mean(dim="time")

    return darray


def compute(dset):
    """Performs the ACC calculation

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

    xdim = list(set(darray.dims) | set(possible_xdims))
    ydim = list(set(darray.dims) | set(possible_ydims))
    zdim = list(set(darray.dims) | set(possible_zdims))

    # Make sure we have exactly one value per coordinate
    for dim in [xdim, ydim, zdim]:
        assert len(dim) == 1, "Ambiguous coordinates found."

    xdim = xdim[0]
    ydim = ydim[0]
    zdim = zdim[0]

    # If the max latitude is positive (i.e. Northern Hemisphere), assume
    # we have a global array that needs to be subset. Otherwise, just keep
    # the array as-is

    darray = (
        darray.sel({xdim: -70, ydim: slice(-70, -54)}, method="nearest")
        if max(darray[ydim]) > 0.0
        else darray
    )

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


def run():
    return


def parse_and_run():
    return
