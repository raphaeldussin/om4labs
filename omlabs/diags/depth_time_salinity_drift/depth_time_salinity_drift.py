#!/usr/bin/env python3

""" Practical Salinity Drift Diagnostic """

from om4labs.om4parser import default_diag_parser
from om4labs.diags.generic_depth_time_drift import read
from om4labs.diags.generic_depth_time_drift import calculate
from om4labs.diags.generic_depth_time_drift import plot
from om4labs.diags.generic_depth_time_drift import run


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

    description = "Diagnostic for layer mean practical salinity drift"

    parser = default_diag_parser(
        description=description,
        template=template,
        exclude=["gridspec", "static", "basin", "obsfile", "hgrid", "topog"],
    )

    parser.add_argument(
        "--description",
        type=str,
        required=False,
        default="Layer-average salinity drift [psu] vs. time",
        help="description string in subtitle",
    )
    parser.add_argument(
        "--varname",
        type=str,
        required=False,
        default="so_xyave",
        help="variable name to plot",
    )
    parser.add_argument(
        "--range",
        type=float,
        required=False,
        default=0.1,
        help="min/max data range for contouring",
    )
    parser.add_argument(
        "--interval", type=float, required=False, default=0.01, help="contour interval"
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


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
