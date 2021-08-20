def parse():
    return


def read(dictArgs):
    dset = xr.open_mfdataset(dictArgs["infile"])
    return dset


def ACC_Transport(varname="umo", zdim="z_l", ydim="yh"):
    darray = dset[varname]
    darray = darray.sum(dim=(zdim, ydim))
    darray = darray * (1 / 1035) * 1.0e-6
    darray = darray.groupby("time.year").mean(dim="time")

    return darray


def compute(dset):
    darray = dset.umo
    if darray.dims[2] == "yh":
        dset = darray.sel(xq=-70, method="nearest", yh=slice(-70, -54))
        result = ACC_Transport(varname="umo", zdim=darray.dims[2], ydim=darray.dims[1])
    else:
        result = ACC_Transport(varname="umo", zdim=darray.dims[2], ydim=darray.dims[1])
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
