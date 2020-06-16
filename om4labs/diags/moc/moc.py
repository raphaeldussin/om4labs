#!/usr/bin/env python3

import argparse
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import m6plot
import palettable
import xarray as xr
import warnings

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


def get_z(ds, depth, var_name):
    """Returns 3d interface positions from netcdf group rg, based on dimension data for variable var_name"""
    if "e" in ds.variables:  # First try native approach
        if len(ds.e) == 3:
            return ds.e
        elif len(ds.e) == 4:
            return ds.e[0]
    if var_name not in ds.variables:
        raise Exception('Variable "' + var_name + '" not found in dataset')
    if len(ds[var_name].shape) < 3:
        raise Exception('Variable "' + var_name + '" must have 3 or more dimensions')
    vdim = ds[var_name].dims[-3]
    if vdim not in ds.variables:
        raise Exception(
            'Variable "' + vdim + '" should be a [CF] dimension variable but is missing'
        )
    if "edges" in ds[vdim].attrs.keys():
        zvar = ds[vdim].edges
    elif "zw" in ds.variables:
        zvar = "zw"
    else:
        raise Exception(
            'Cannot figure out vertical coordinate from variable "' + var_name + '"'
        )
    if not len(ds[zvar].shape) == 1:
        raise Exception('Variable "' + zvar + '" was expected to be 1d')
    zw = np.array(ds[zvar][:])
    Zmod = np.zeros((zw.shape[0], depth.shape[0], depth.shape[1]))
    for k in range(zw.shape[0]):
        Zmod[k] = -np.minimum(depth, abs(zw[k]))
    return Zmod


def compute_msftyyz(vmo, basin_code, nc_misval=1.0e20):
    """Computes meridional overturning streamfunction according to CMIP6 spec"""
    atlantic_arctic_mask = generate_basin_masks(basin_code, basin="atlantic_arctic")
    indo_pacific_mask = generate_basin_masks(basin_code, basin="indo_pacific")
    msftyyz = np.ma.ones((vmo.shape[0], 3, vmo.shape[1] + 1, vmo.shape[2])) * 0.0
    msftyyz[:, 0, :, :] = moc_maskedarray(vmo, mask=atlantic_arctic_mask)
    msftyyz[:, 1, :, :] = moc_maskedarray(vmo, mask=indo_pacific_mask)
    msftyyz[:, 2, :, :] = moc_maskedarray(vmo)
    msftyyz.long_name = "Ocean Y Overturning Mass Streamfunction"
    msftyyz.units = "kg s-1"
    msftyyz.coordinates = "region"
    msftyyz.cell_methods = "z_i:point yq:point time:mean"
    msftyyz.time_avg_info = "average_T1,average_T2,average_DT"
    msftyyz.standard_name = "ocean_y_overturning_mass_streamfunction"
    return msftyyz


def moc_maskedarray(vh, mask=None):
    """Computes the overturning streamfunction given a masked array"""
    if mask is not None:
        _mask = np.ma.masked_where(np.not_equal(mask, 1.0), mask)
    else:
        _mask = 1.0
    _vh = vh * _mask
    _vh_btm = np.ma.expand_dims(_vh[:, -1, :, :] * 0.0, axis=1)
    _vh = np.ma.concatenate((_vh, _vh_btm), axis=1)
    _vh = np.ma.sum(_vh, axis=-1) * -1.0
    _vh = _vh[:, ::-1]  # flip z-axis so running sum is from ocean floor to surface
    _vh = np.ma.cumsum(_vh, axis=1)
    _vh = _vh[:, ::-1]  # flip z-axis back to original order
    return _vh


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


def arguments(cliargs=None):
    """
    Function to capture the user-specified command line options
    """
    description = """ """

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-b", "--basin", type=str, default=None, help="Path to basin code file"
    )

    parser.add_argument(
        "-g", "--gridspec", type=str, default=None, help="Path to gridspec file"
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

    return parser.parse_args(cliargs)


def read(infile, basin, gridspec, topog, varname="vmo"):
    """MOC plotting script"""

    print("Reading input data")
    ds = xr.open_mfdataset(infile, combine="by_coords")
    ds_basin = xr.open_dataset(basin)
    ds_topog = xr.open_dataset(topog)
    ds_gridspec = xr.open_dataset(gridspec)

    # horizontal grid
    x = np.array(ds_gridspec.x.to_masked_array())[::2, ::2]
    y = np.array(ds_gridspec.y.to_masked_array())[::2, ::2]

    # depth coordinate
    depth = ds_topog.deptho.to_masked_array()
    depth = np.where(np.isnan(depth), 0.0, depth)
    z = get_z(ds, depth, varname)

    # basin code
    basin_code = ds_basin.basin.to_masked_array()

    # vmo
    vmo = ds[varname].to_masked_array()

    return x, y, z, basin_code, vmo


def calculate(vmo, basin_code):
    """Main computational script"""

    print("Calculating")
    msftyyz = compute_msftyyz(vmo, basin_code)

    return msftyyz


def plot(y, z, msftyyz, label=None):
    """Plotting script"""

    def _findExtrema(
        ax, y, z, psi, min_lat=-90.0, max_lat=90.0, min_depth=0.0, mult=1.0
    ):
        """Function to annotate the max/min values on the MOC plot"""
        psiMax = mult * np.amax(
            mult * np.ma.array(psi)[(y >= min_lat) & (y <= max_lat) & (z < -min_depth)]
        )
        idx = np.argmin(np.abs(psi - psiMax))
        (j, i) = np.unravel_index(idx, psi.shape)
        ax.plot(y[j, i], z[j, i], "kx")
        ax.text(y[j, i], z[j, i], "%.1f" % (psi[j, i]))

    def _plotPsi(y, z, psi, ci, title, cmap=None):
        """Function to plot zonal mean streamfunction"""
        plt.contourf(y, z, psi, levels=ci, cmap=cmap, extend="both")
        cbar = plt.colorbar()
        plt.contour(y, z, psi, levels=ci, colors="k")
        plt.gca().set_yscale("splitscale", zval=[0, -2000, -6500])
        plt.title(title)
        cbar.set_label("[Sv]")
        plt.ylabel("Elevation [m]")

    print("Making Plot.")
    z = z.min(axis=-1)
    yy = y[1:, :].max(axis=-1) + 0 * z

    psi = msftyyz * 1.0e-9

    ci = m6plot.formatting.pmCI(0.0, 40.0, 5.0)
    cmap = palettable.cmocean.diverging.Balance_20.get_mpl_colormap()

    fig = plt.figure(figsize=(8.5, 11))
    ax1 = plt.subplot(3, 1, 1)
    psiPlot = psi[0, 0]
    _plotPsi(yy, z, psiPlot, ci, "Atlantic MOC [Sv]", cmap=cmap)
    _findExtrema(ax1, yy, z, psiPlot, min_lat=26.5, max_lat=27.0)
    _findExtrema(ax1, yy, z, psiPlot, max_lat=-33.0)
    _findExtrema(ax1, yy, z, psiPlot)

    ax2 = plt.subplot(3, 1, 2)
    psiPlot = psi[0, 1]
    _plotPsi(yy, z, psiPlot, ci, "Indo-Pacific MOC [Sv]", cmap=cmap)
    _findExtrema(ax2, yy, z, psiPlot, min_depth=2000.0, mult=-1.0)
    _findExtrema(ax2, yy, z, psiPlot)

    ax3 = plt.subplot(3, 1, 3)
    psiPlot = psi[0, 2]
    _plotPsi(yy, z, psiPlot, ci, "Global MOC [Sv]", cmap=cmap)
    _findExtrema(ax3, yy, z, psiPlot, max_lat=-30.0)
    _findExtrema(ax3, yy, z, psiPlot, min_lat=25.0)
    _findExtrema(ax3, yy, z, psiPlot, min_depth=2000.0, mult=-1.0)
    plt.xlabel(r"Latitude [$\degree$N]")

    plt.subplots_adjust(hspace=0.2)

    if label is not None:
        plt.suptitle(label)

    return fig


def run(args):
    """Function to call read, calc, and plot in sequence"""

    # set visual backend
    if args["interactive"] is False:
        plt.switch_backend("Agg")
    else:
        plt.switch_backend("TkAgg")

    print(f"Matplotlib is using the {mpl.get_backend()} back-end.")

    # --- the main show ---
    x, y, z, basin_code, vmo = read(
        args["infile"], args["basin"], args["gridspec"], args["topog"]
    )
    msftyyz = calculate(vmo, basin_code)
    fig = plot(y, z, msftyyz, args["label"])
    # ---------------------

    # do something with the figure
    if args["interactive"] is True:
        plt.show(fig)
    else:
        imgbuf = io.BytesIO()
        fig.savefig(imgbuf, format=args["format"], dpi=150, bbox_inches="tight")
        with open(f"{args['outdir']}/mocfig.{args['format']}", "wb") as f:
            f.write(imgbuf.getbuffer())


def parse_and_run(cliargs=None):
    args = arguments(cliargs)
    args = args.__dict__
    run(args)


if __name__ == "__main__":
    parse_and_run()
