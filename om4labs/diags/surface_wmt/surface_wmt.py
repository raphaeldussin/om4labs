#!/usr/bin/env python3

import argparse
import pkg_resources as pkgr
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from om4labs import m6plot
import palettable
import xarray as xr
import warnings

### To be replaced when API of xwmt is confirmed
from wmt_inert_tracer.swmt import swmt
from wmt_inert_tracer.compute import lbin_define
from wmt_inert_tracer.preprocessing import preprocessing

from om4labs.om4common import horizontal_grid
from om4labs.om4common import image_handler
from om4labs.om4common import date_range
from om4labs.om4common import open_intake_catalog
from om4labs.om4parser import default_diag_parser

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


def calculate(ds, bins, group_tend):
    """Calculates watermass transformation from surface fluxes"""
    
    # Think it makes most sense here to not group tendencies
    G = swmt(ds).G('sigma0', bins=bins, group_tend=group_tend)
    
    # If tendencies were grouped then G is a DataArray
    # For consistency in plotting function, convert it to a dataset
    if group_tend:
        G = G.to_dataset()
        
    return G


def parse(cliargs=None, template=False):
    """
    Function to capture the user-specified command line options
    """
    description = """ """

    parser = default_diag_parser(
        description=description, template=template, exclude=["obsfile", "topog"]
    )
    
    parser.add_argument(
        "--bins", type=str, default="20,30,0.1", help="Density bins at which to evaluate transformation, provided as start, stop, increment.",
    )
    
    parser.add_argument(
        "--group_tend", dest="group_tend", action="store_true", help="Group heat and salt tendencies together, i.e. only return the total transformation. Not passing this could lead to a performance cost.",
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs, heatflux_varname="hfds", saltflux_varname="sfdsi", 
         fwflux_varname="wfo", sst_varname="tos", sss_varname="sos"):
    """Read in surface flux data"""

    infile = dictArgs["infile"]
    ds = xr.open_mfdataset(infile, combine="by_coords", use_cftime=True)

    ### WMT preprocessing step
    # Perhaps we should pull out some of what happens in here
    ds = preprocessing(ds, grid=ds, decode_times=False, verbose=False)
    
    # Check for presence of correct variables
    # Get gridcell area ## How to specify to get from static grid?
    # Possibly get some masks ?

    ds["areacello"] = xr.open_mfdataset(dictArgs["static"])["areacello"]
    
    if "bins" in dictArgs:
        bins_args = dictArgs["bins"]
        bins_args = tuple([float(x) for x in bins_args.split(",")])
        bins = np.arange(*bins_args)
    else:
        # Default bins
        bins = np.arange(20,30,0.1)
    # Expand bins to capture full density range
    bins = np.concatenate((np.array([0]),bins,np.array([100])))
    
    # Retrieve group_tend boolean
    group_tend=dictArgs["group_tend"]

    return (
        ds,
        bins,
        group_tend
    )

def plot(G):

    # Don't plot first or last bin (expanded to capture full range)
    G = G.isel(sigma0=slice(1,-1))
    levs = G['sigma0'].values
    
    # Take annual mean and load
    G = G.mean('time').load()
    # Get terms in dataset
    terms = list(G.data_vars)
    
    fig,ax = plt.subplots()
    # Plot each term
    for term in terms:
        if term =='heat':
            color='tab:red'
        elif term =='salt':
            color='tab:blue'
        else:
            color='k'
        ax.plot(levs,G[term],label=term,color=color)
        
    # If terms were not grouped then sum them up to get total
    if len(terms)>1:
        total = xr.zeros_like(G[terms[0]])
        for term in terms:
            total += G[term]
        ax.plot(levs,total,label='total',color='k')
        
    ax.legend()
    ax.set_xlabel('SIGMA0')
    ax.set_ylabel('TRANSFORMATION ($m^3s^{-1}$)')
    ax.autoscale(enable=True, axis='x', tight=True)

    return fig


def run(dictArgs):
    """Function to call read, calc, and plot in sequence"""

    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")

    # --- the main show ---
    (
        ds,
        bins
    ) = read(dictArgs)

    G = calculate(ds,bins)

    fig = plot(G)

    filename = f"{dictArgs['outdir']}/surface_wmt"
    imgbufs = image_handler([fig], dictArgs, filename=filename)

    return imgbufs


def parse_and_run(cliargs=None):
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
