#!/usr/bin/env python3

import argparse
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from om4labs import m6plot
import xarray as xr

from om4labs.om4common import image_handler
from om4labs.om4parser import default_diag_parser


def parse(cliargs=None, template=False):
    """
    Function to capture the user-specified command line options
    """
    description = """Generic routine for plotting section transports"""

    exclude = [
        "basin",
        "config",
        "gridspec",
        "hgrid",
        "obsfile",
        "platform",
        "static",
        "suptitle",
        "topog",
    ]

    parser = default_diag_parser(
        description=description, template=template, exclude=exclude
    )

    parser.add_argument(
        "--passage_label",
        type=str,
        default="",
        help="Text to use for the name of the passage being plotted",
    )

    parser.add_argument(
        "--obsrange",
        type=str,
        default=None,
        help="Comma-separated tuple of min and max observed ranges (e.g. -20,20)",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(infile, varlist=["umo", "vmo"]):
    """Reads input files from a working directory"""

    ds = xr.open_mfdataset(infile, use_cftime=True, combine="by_coords")
    assert any(
        field in ds.variables for field in varlist
    ), f"Input dataset must include one of {varlist}"
    return ds


def calculate(
    dset_transport,
    components=["umo", "vmo"],
    zlim=None,
    conversion=1.0e-9,
    monthavg=True,
):
    """Function to calculate the transport normal to defined passage"""

    # cross-check variable list with input dataset
    complist = [
        varname for varname in components if varname in dset_transport.variables
    ]

    # transport will eventually contain the total transport, from both
    # u and v components if provided. It is initialized here as a
    # list of xarray.DataArrays()
    transport = []

    # loop over transport components
    for comp in complist:

        # obtain a list of dimensions
        component = dset_transport[comp]
        dims = list(component.dims)

        # detect x and y dimensions for this DataArray
        xdim = [dim for dim in dims if dim.startswith("x")]
        ydim = [dim for dim in dims if dim.startswith("y")]

        if len(xdim) != 1:
            raise RuntimeError(f"Unable to find x dimension in {comp}")
        if len(ydim) != 1:
            raise RuntimeError(f"Unable to find y dimension in {comp}")

        xdim = xdim[0]
        ydim = ydim[0]

        # detect and sum the transport along the dimension normal to the passage definition
        avedim = ydim if len(dset_transport[ydim]) > len(dset_transport[xdim]) else xdim
        component = component.sum(dim=avedim)

        # ensure we have a 3-d data cube at this step.
        assert len(component.dims) == 3, f"{comp} may be missing a time coordinate"

        # The data cube should also contain a singleton dimension at this step
        component = component.squeeze()
        assert len(component.dims) == 2, f"{comp} did not reduce to (time,depth)"

        # Append this array to the total transport
        transport.append(component)

    # concatentate components along a new axis and take the sum
    transport = xr.concat(transport, dim="component")
    transport = transport.sum(dim="component")
    assert (
        len(transport.dims) == 2
    ), f"Total transport has wrong dimensions, should be (time,depth)"

    # restrict vertical range if asked
    zcoord = transport.dims[1]
    if zlim is not None:
        assert isinstance(zlim, tuple), "zlim must be a tuple of depths"
        zlim = {zcoord: slice(*zlim)}
        transport = transport.sel(**zlim)

    # sum in the verical
    transport = transport.sum(dim=zcoord)

    # check that we have a timeseries
    assert len(transport.dims) == 1, "Dataset did not yield a timeseries"

    # apply conversion factor
    transport = transport * conversion

    # create montly average if asked
    transport = transport.resample(time="1M").mean() if monthavg else transport

    # rename the output field
    transport = transport.rename("summed transport")

    # load array
    transport.load()

    return transport


def plot(transport, ylim=None, label=None, passage_label=None, obsrange=None):
    fig = plt.figure(figsize=(8.5, 4))
    ax = plt.subplot(1, 1, 1)

    # plot obs range
    if obsrange is not None:
        minval = min(obsrange) * np.ones((len(transport.values)))
        maxval = max(obsrange) * np.ones((len(transport.values)))
        plt.fill_between(
            transport.time.values,
            minval,
            maxval,
            color="black",
            alpha=0.2,
            edgecolor=None,
        )

    ax.plot(transport.time, transport.values, color="k")

    ax.text(0.02, 1.02, passage_label, ha="left", fontsize=16, transform=ax.transAxes)
    ax.text(0.98, 1.02, label, ha="right", fontsize=12, transform=ax.transAxes)

    ax.set_ylim(ylim)

    return fig


def run(dictArgs):
    """Function to call read, calc, and plot in sequence"""

    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")
    else:
        plt.switch_backend("TkAgg")

    # --- the main show ---

    dset_transport = read(dictArgs["infile"])
    transport = calculate(dset_transport)

    dictArgs["obsrange"] = (
        tuple([float(x) for x in dictArgs["obsrange"].split(",")])
        if dictArgs["obsrange"] is not None
        else None
    )

    fig = plot(
        transport,
        label=dictArgs["label"],
        passage_label=dictArgs["passage_label"],
        obsrange=dictArgs["obsrange"],
    )

    # ---------------------

    # construct output filename based on "passage_label" argument, if present
    filename = dictArgs["passage_label"]
    filename = filename.replace(" ", "_") if filename != "" else "section"
    filename = f"{dictArgs['outdir']}/{filename}"
    imgbufs = image_handler([fig], dictArgs, filename=filename)

    return imgbufs


def parse_and_run(cliargs=None):
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
