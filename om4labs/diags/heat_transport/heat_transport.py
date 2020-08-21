#!/usr/bin/env python3

import argparse
import pkg_resources as pkgr
import intake
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from om4labs import m6plot
import palettable
import xarray as xr
import warnings

from om4labs.om4common import image_handler
from om4labs.om4common import DefaultDictParser

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


def generate_basin_masks(basin_code, basin=None):
    """Function to generate pre-defined basin masks"""
    mask = basin_code * 0
    if basin == "atlantic_arctic":
        mask[
            (basin_code == 2)
            | (basin_code == 4)
            | (basin_code == 6)
            | (basin_code == 7)
            | (basin_code == 8)
        ] = 1.0
    elif basin == "indo_pacific":
        mask[(basin_code == 3) | (basin_code == 5)] = 1.0
    elif isinstance(basin, int):
        mask[(basin_code == basin)] = 1.0
    else:
        mask[(basin_code >= 1)] = 1.0
    return mask

def compute(advective, diffusive=None, vmask=None, rho0=1.035e3, Cp=3989.):
    """Converts vertically integrated temperature advection into heat transport"""

    if diffusive is not None:
      HT = advective + diffusive
    else:
      HT = advective
    HT = HT.mean(dim="time")

    if advective.units == "Celsius meter3 second-1":
      HT = HT * (rho0 * Cp)
      HT = HT * 1.e-15  # convert to PW
    elif advective.units == "W m-2":  # bug in MOM6 units (issue #934), keep for retrocompatibility
      HT = HT * 1.e-15
    elif advective.units == "W":
      HT = HT * 1.e-15
    else:
      print('Unknown units')

    HT = HT.to_masked_array()

    if vmask is not None:
        HT = HT*vmask

    HT = HT.sum(axis=-1)
    HT = HT.squeeze() # sum in x-direction

    return HT


def parse(cliargs=None, template=False):
    """
    Function to capture the user-specified command line options
    """
    description = """ """

    if template is True:
        parser = DefaultDictParser(
            description=description, formatter_class=argparse.RawTextHelpFormatter
        )
    else:
        parser = argparse.ArgumentParser(
            description=description, formatter_class=argparse.RawTextHelpFormatter
        )

    parser.add_argument(
        "-b", "--basin", type=str, default=None, help="Path to basin code file"
    )

    parser.add_argument(
        "-g", "--gridspec", type=str, default=None, help="Path to gridspec file"
    )

    parser.add_argument("-m", "--model", type=str, default=None, help="Model Class")

    parser.add_argument(
        "--platform",
        type=str,
        required=False,
        default="gfdl",
        help="computing platform, default is gfdl",
    )

    parser.add_argument(
        "infile",
        metavar="INFILE",
        type=str,
        nargs="+",
        help="Path to input NetCDF file(s)",
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode displays plot to screen. Default is False",
    )

    parser.add_argument(
        "-l", "--label", type=str, default="", help="String label to annotate the plot"
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
        "-t", "--topog", type=str, default=None, help="Path to topog file"
    )

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs, adv_varname="T_ady_2d", dif_varname="T_diffy_2d"):
    """Read in heat transport data"""

    infile = dictArgs["infile"]

    if dictArgs["model"] is not None:
        # use dataset from catalog, either from command line or default
        cat_platform = (
            f"catalogs/{dictArgs['model']}_catalog_{dictArgs['platform']}.yml"
        )
        catfile = pkgr.resource_filename("om4labs", cat_platform)
        cat = intake.open_catalog(catfile)
        ds_basin = cat["basin"].to_dask()
        ds_gridspec = cat["ocean_hgrid"].to_dask()

    if dictArgs["basin"] is not None:
        ds_basin = xr.open_dataset(dictArgs["basin"])

    ds = xr.open_mfdataset(infile, combine="by_coords")

    # horizontal grid
    x = np.array(ds_gridspec.x.to_masked_array())[::2, ::2]
    y = np.array(ds_gridspec.y.to_masked_array())[::2, ::2]

    # nominal y coordinate
    yq = ds.yq.to_masked_array()

    # basin code
    basin_code = ds_basin.basin.to_masked_array()

    # basin masks
    atlantic_arctic_mask = generate_basin_masks(basin_code, basin="atlantic_arctic")
    indo_pacific_mask = generate_basin_masks(basin_code, basin="indo_pacific")

    # advective component of transport
    advective = ds[adv_varname]

    # diffusive component of transport
    if dif_varname in ds.variables:
        diffusive = ds[dif_varname]
    else:
        diffusive = None

    return x, y, yq, basin_code, atlantic_arctic_mask, indo_pacific_mask, advective, diffusive


def calculate(advective, diffusive=None, basin_code=None):
    """Main computational script"""

    msftyyz = compute_msftyyz(vmo, basin_code)

    return msftyyz

class GWObs:
    class _gw:
        def __init__(self,lat,trans,err):
            self.lat = lat
            self.trans = trans
            self.err = err
            self.minerr = trans - err
            self.maxerr = trans + err
        def annotate(self,ax):
            for n in range(0,len(self.minerr)):
                if n == 0:
                    ax.plot([self.lat[n],self.lat[n]], [self.minerr[n],self.maxerr[n]], 'c', linewidth=2.0, label='G&W')
                else:
                    ax.plot([self.lat[n],self.lat[n]], [self.minerr[n],self.maxerr[n]], 'c', linewidth=2.0)
                ax.scatter(self.lat,self.trans,marker='s',facecolor='cyan')

    def __init__(self):
        self.gbl = self._gw(
                       np.array([-30., -19., 24., 47.]),
                       np.array([-0.6, -0.8, 1.8, 0.6]),
                       np.array([0.3, 0.6, 0.3, 0.1]),
                       )
        self.atl = self._gw(
                       np.array([-45., -30., -19., -11., -4.5, 7.5, 24., 47.]),
                       np.array([0.66, 0.35, 0.77, 0.9, 1., 1.26, 1.27, 0.6]),
                       np.array([0.12, 0.15, 0.2, 0.4, 0.55, 0.31, 0.15, 0.09]),
                       )
        self.indpac = self._gw(
                       np.array([-30., -18., 24., 47.]),
                       np.array([-0.9, -1.6, 0.52, 0.]),
                       np.array([0.3, 0.6, 0.2, 0.05,]),
                       )

def plot(dictArgs,yq,trans_global,trans_atlantic,trans_pacific):

    # Load observations for plotting
    GW = GWObs()

    cat_platform = "catalogs/obs_catalog_" + dictArgs["platform"] + ".yml"
    catfile = pkgr.resource_filename("om4labs", cat_platform)
    cat = intake.open_catalog(catfile)
    fObs = cat["Trenberth_and_Caron"].to_dask()

    yobs = fObs.ylat.to_masked_array()
    NCEP_Global = fObs.OTn.to_masked_array()
    NCEP_Atlantic = fObs.ATLn.to_masked_array()
    NCEP_IndoPac = fObs.INDPACn.to_masked_array()
    ECMWF_Global = fObs.OTe.to_masked_array()
    ECMWF_Atlantic = fObs.ATLe.to_masked_array()
    ECMWF_IndoPac = fObs.INDPACe.to_masked_array()

    fig = plt.figure(figsize=(6,10))
    
    # Global Heat Transport
    ax1 = plt.subplot(3,1,1)
    plt.plot(yq, yq*0., 'k', linewidth=0.5)
    plt.plot(yq, trans_global, 'r', linewidth=1.5,label='Model')
    GW.gbl.annotate(ax1)
    plt.plot(yobs,NCEP_Global,'k--',linewidth=0.5,label='NCEP')
    plt.plot(yobs,ECMWF_Global,'k.',linewidth=0.5,label='ECMWF')
    plt.ylim(-2.5,3.0)
    plt.grid(True)
    plt.legend(loc=2,fontsize=10)
    ax1.text(0.01,1.02,'a. Global Poleward Heat Transport',ha='left',transform=ax1.transAxes)

    #if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')

    # Atlantic Heat Transport
    ax2 = plt.subplot(3,1,2)
    plt.plot(yq, yq*0., 'k', linewidth=0.5)
    trans_atlantic[yq<-34] = np.nan
    plt.plot(yq, trans_atlantic, 'r', linewidth=1.5,label='Model')
    GW.atl.annotate(ax2)
    plt.plot(yobs,NCEP_Atlantic,'k--',linewidth=0.5,label='NCEP')
    plt.plot(yobs,ECMWF_Atlantic,'k.',linewidth=0.5,label='ECMWF')
    plt.ylim(-0.5,2.0)
    plt.grid(True)
    ax2.text(0.01,1.02,'b. Atlantic Poleward Heat Transport',ha='left',transform=ax2.transAxes)

    # Indo-pacific Heat Transport
    ax3 = plt.subplot(3,1,3)
    plt.plot(yq, yq*0., 'k', linewidth=0.5)
    trans_pacific[yq<-34] = np.nan
    plt.plot(yq, trans_pacific, 'r', linewidth=1.5,label='Model')
    GW.indpac.annotate(ax3)
    plt.plot(yobs,NCEP_IndoPac,'k--',linewidth=0.5,label='NCEP')
    plt.plot(yobs,ECMWF_IndoPac,'k.',linewidth=0.5,label='ECMWF')
    plt.ylim(-2.5,1.5)
    plt.grid(True)
    plt.xlabel(r'Latitude [$\degree$N]')
    ax3.text(0.01,1.02,'c. Indo-Pacific Poleward Heat Transport',ha='left',transform=ax3.transAxes)

    for ax in [ax1,ax2]:
        ax.set_xticklabels([])

    plt.subplots_adjust(hspace=0.3)

    # Annotations
    fig.text(0.05,0.05,r"Trenberth, K. E. and J. M. Caron, 2001: Estimates of Meridional Atmosphere and Ocean Heat Transports. J.Climate, 14, 3433-3443.", fontsize=6)
    fig.text(0.05,0.04,r"Ganachaud, A. and C. Wunsch, 2000: Improved estimates of global ocean circulation, heat transport and mixing from hydrographic data.", fontsize=6)
    fig.text(0.05,0.03,r"Nature, 408, 453-457", fontsize=6)

    if dictArgs['label'] is not None:
        plt.suptitle(dictArgs['label'])

    #HTplot = heatTrans(advective, diffusive, vmask=m*numpy.roll(m,-1,axis=-2))

    return fig 

def run(dictArgs):
    """Function to call read, calc, and plot in sequence"""

    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")
    else:
        # plt.switch_backend("TkAgg")
        plt.switch_backend("qt5agg")

    # --- the main show ---
    x, y, yq, basin_code, atlantic_arctic_mask, indo_pacific_mask, advective, diffusive = read(dictArgs)
   
    trans_global   = compute(advective, diffusive)
    trans_atlantic = compute(advective, diffusive, vmask=atlantic_arctic_mask)
    trans_pacific  = compute(advective, diffusive, vmask=indo_pacific_mask)

    fig = plot(dictArgs,yq,trans_global,trans_atlantic,trans_pacific)
 
    filename = f"{dictArgs['outdir']}/heat_transport"
    imgbufs = image_handler([fig], dictArgs, filename=filename)

    return imgbufs


def parse_and_run(cliargs=None):
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
