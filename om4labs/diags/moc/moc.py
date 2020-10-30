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
from om4labs.om4parser import default_diag_parser

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


def parse(cliargs=None, template=False):
    """
    Function to capture the user-specified command line options
    """
    description = """ """

    parser = default_diag_parser(description=description, template=template)

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs, varname="vmo"):
    """MOC plotting script"""

    infile = dictArgs["infile"]

    if dictArgs["config"] is not None:
        # use dataset from catalog, either from command line or default
        cat_platform = (
            f"catalogs/{dictArgs['config']}_catalog_{dictArgs['platform']}.yml"
        )
        catfile = pkgr.resource_filename("om4labs", cat_platform)
        cat = intake.open_catalog(catfile)
        ds_basin = cat["basin"].to_dask()
        ds_topog = cat["topog"].to_dask()
        ds_gridspec = cat["ocean_hgrid"].to_dask()

    if dictArgs["basin"] is not None:
        ds_basin = xr.open_dataset(dictArgs["basin"])
    if dictArgs["gridspec"] is not None:
        ds_topog = xr.open_dataset(dictArgs["topog"])
    if dictArgs["topog"] is not None:
        ds_gridspec = xr.open_dataset(dictArgs["gridspec"])

    ds = xr.open_mfdataset(infile, combine="by_coords")

    # horizontal grid
    x = np.array(ds_gridspec.x.to_masked_array())[::2, ::2]
    y = np.array(ds_gridspec.y.to_masked_array())[::2, ::2]

    # depth coordinate
    if "deptho" in list(ds_topog.variables):
        depth = ds_topog.deptho.to_masked_array()
    elif "depth" in list(ds_topog.variables):
        depth = ds_topog.depth.to_masked_array()
    else:
        raise ValueError("Unable to find depth field.")
    depth = np.where(np.isnan(depth), 0.0, depth)
    if varname == "msftyyz":
        zw = np.array(ds["z_i"][:])
        Zmod = np.zeros((zw.shape[0], depth.shape[0], depth.shape[1]))
        for k in range(zw.shape[0]):
            Zmod[k] = -np.minimum(depth, abs(zw[k]))
        z = Zmod
    else:
        z = get_z(ds, depth, varname)

    # t-cell nominal y coordinate
    if "yh" in list(ds_topog.variables):
        yh = ds_topog.yh.to_masked_array()
    else:
        yh = ds.yh.to_masked_array()

    # basin code
    basin_code = ds_basin.basin.to_masked_array()

    # basin masks
    atlantic_arctic_mask = generate_basin_masks(basin_code, basin="atlantic_arctic")
    indo_pacific_mask = generate_basin_masks(basin_code, basin="indo_pacific")

    # vmo
    arr = ds[varname].to_masked_array()

    return x, y, yh, z, depth, basin_code, atlantic_arctic_mask, indo_pacific_mask, arr


def calculate(vmo, basin_code):
    """Main computational script"""

    msftyyz = compute_msftyyz(vmo, basin_code)

    return msftyyz


def plot(y, yh, z, depth, atlantic_arctic_mask, indo_pacific_mask, msftyyz, label=None):
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

    def _plotPsi(
        y, z, psi, ci, title, cmap=None, xlim=None, topomask=None, yh=None, zz=None
    ):
        """Function to plot zonal mean streamfunction"""
        if topomask is not None:
            psi = np.array(np.where(psi.mask, 0.0, psi))
        else:
            psi = np.array(np.where(psi.mask, np.nan, psi))
        plt.contourf(y, z, psi, levels=ci, cmap=cmap, extend="both")
        cbar = plt.colorbar()
        plt.contour(y, z, psi, levels=ci, colors="k", linewidths=0.4)
        plt.contour(y, z, psi, levels=[0], colors="k", linewidths=0.8)
        if topomask is not None:
            cMap = mpl.colors.ListedColormap(["gray"])
            plt.pcolormesh(yh, -1.0 * zz, topomask, cmap=cMap)
        plt.gca().set_yscale("splitscale", zval=[0, -2000, -6500])
        plt.title(title)
        if xlim is not None:
            plt.gca().set_xlim(xlim)
        cbar.set_label("[Sv]")
        plt.ylabel("Elevation [m]")

    def _create_topomask(depth, yh, mask=None):
        if mask is not None:
            depth = np.where(mask == 1, depth, 0.0)
        topomask = depth.max(axis=-1)
        _y = yh
        _z = np.arange(0, 7100, 100)
        _yy, _zz = np.meshgrid(_y, _z)
        topomask = np.tile(topomask[None, :], (len(_z), 1))
        topomask = np.ma.masked_where(_zz < topomask, topomask)
        topomask = topomask * 0.0
        return topomask, _z

    if len(z.shape) != 1:
        z = z.min(axis=-1)
    yy = y[1:, :].max(axis=-1) + 0 * z

    psi = msftyyz * 1.0e-9

    atlantic_topomask, zz = _create_topomask(depth, yh, atlantic_arctic_mask)
    indo_pacific_topomask, zz = _create_topomask(depth, yh, indo_pacific_mask)
    global_topomask, zz = _create_topomask(depth, yh)

    ci = m6plot.formatting.pmCI(0.0, 43.0, 3.0)
    cmap = palettable.cmocean.diverging.Balance_20.get_mpl_colormap()

    fig = plt.figure(figsize=(8.5, 11))
    ax1 = plt.subplot(3, 1, 1, facecolor="gray")
    psiPlot = psi[0, 0]
    _plotPsi(
        yy,
        z,
        psiPlot,
        ci,
        "Atlantic MOC [Sv]",
        cmap=cmap,
        xlim=(-40, 90),
        topomask=atlantic_topomask,
        yh=yh,
        zz=zz,
    )
    _findExtrema(ax1, yy, z, psiPlot, min_lat=26.5, max_lat=27.0)
    _findExtrema(ax1, yy, z, psiPlot, max_lat=-33.0)
    _findExtrema(ax1, yy, z, psiPlot)

    ax2 = plt.subplot(3, 1, 2, facecolor="gray")
    psiPlot = psi[0, 1]
    _plotPsi(
        yy,
        z,
        psiPlot,
        ci,
        "Indo-Pacific MOC [Sv]",
        cmap=cmap,
        xlim=(-40, 65),
        topomask=indo_pacific_topomask,
        yh=yh,
        zz=zz,
    )
    _findExtrema(ax2, yy, z, psiPlot, min_depth=2000.0, mult=-1.0)
    _findExtrema(ax2, yy, z, psiPlot)

    ax3 = plt.subplot(3, 1, 3, facecolor="gray")
    psiPlot = psi[0, 2]
    _plotPsi(
        yy,
        z,
        psiPlot,
        ci,
        "Global MOC [Sv]",
        cmap=cmap,
        topomask=global_topomask,
        yh=yh,
        zz=zz,
    )
    _findExtrema(ax3, yy, z, psiPlot, max_lat=-30.0)
    _findExtrema(ax3, yy, z, psiPlot, min_lat=25.0)
    _findExtrema(ax3, yy, z, psiPlot, min_depth=2000.0, mult=-1.0)
    plt.xlabel(r"Latitude [$\degree$N]")

    plt.subplots_adjust(hspace=0.2)

    if label is not None:
        plt.suptitle(label)

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
    ds = xr.open_mfdataset(dictArgs["infile"], combine="by_coords")
    if "msftyyz" in list(ds.variables):
        varname = "msftyyz"
    elif "vmo" in list(ds.variables):
        varname = "vmo"
    ds.close()

    x, y, yh, z, depth, basin_code, atlantic_arctic_mask, indo_pacific_mask, arr = read(
        dictArgs, varname=varname
    )

    if varname != "msftyyz":
        msftyyz = calculate(arr, basin_code)
    else:
        msftyyz = arr

    fig = plot(
        y,
        yh,
        z,
        depth,
        atlantic_arctic_mask,
        indo_pacific_mask,
        msftyyz,
        dictArgs["label"],
    )
    # ---------------------

    filename = f"{dictArgs['outdir']}/moc"
    imgbufs = image_handler([fig], dictArgs, filename=filename)

    return imgbufs


def parse_and_run(cliargs=None):
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
