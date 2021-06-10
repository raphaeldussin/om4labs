# need a data_read, data_sel, data_reduce

import argparse
import calendar
import glob
import io
import os
import pathlib
import signal
import sys
import tarfile as tf
import warnings
from datetime import datetime

import intake
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources as pkgr
import scipy
import xarray as xr
import xesmf as xe
from cmip_basins import generate_basin_codes
from packaging import version
from xgcm import Grid

try:
    from om4labs.helpers import try_variable_from_list
except ImportError:
    # DORA mode, works without install.
    # reads from current directory
    from helpers import try_variable_from_list

from static_downsampler.static import (
    extend_supergrid_array,
    subsample_supergrid,
    sum_on_supergrid,
)

possible_names = {}
possible_names["lon"] = ["lon", "LON", "longitude", "LONGITUDE"]
possible_names["lat"] = ["lat", "LAT", "latitude", "LATITUDE"]
possible_names["time"] = ["time", "TIME", "latitude"]
possible_names["depth"] = ["z_l", "depth", "DEPTH"]
possible_names["interfaces"] = ["z_i"]


class DefaultDictParser(argparse.ArgumentParser):
    """ argparse extention that bypasses error and returns a dict of defaults """

    def error(self, message):
        actions = self.__dict__["_actions"]
        defaults = {}
        for act in actions[1::]:
            defaults[act.__dict__["dest"]] = act.__dict__["default"]
        return defaults


def date_range(ds, ref_time="1970-01-01T00:00:00Z"):
    """Returns a tuple of start year and end year from xarray dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset

    Returns
    -------
    tuple
        (start year, end year) for time dimension of the dataset
    """

    if "time_bnds" in list(ds.variables):

        # Xarray decodes bounds times relative to the epoch and
        # returns a numpy timedelta object in some instances
        # instead of a cftime datetime object. Manual decoding
        # and shifting may be necessary

        if isinstance(ds["time_bnds"].values[0][0], np.timedelta64):

            # When opening multi-file datasets with open_mfdataset(),
            # xarray strips out the calendar encoding. Since bounds
            # are computed differently to begin with, fall back to
            # another FMS generated time variable to get the calendar
            # base date.

            if "units" in ds["time"].encoding.keys():
                base_time = ds["time"].encoding["units"]
            elif "units" in ds["average_T1"].encoding.keys():
                base_time = ds["average_T1"].encoding["units"]
            else:
                base_time = None

            if base_time is not None:
                base_time = base_time.split(" ")[2:4]
                base_time = np.datetime64(f"{base_time[0]}T{base_time[1]}Z")
                offset = base_time - np.datetime64(ref_time)

                t0 = ds["time_bnds"].values[0][0] + offset
                t0 = datetime.fromtimestamp(int(np.ceil(int(t0) * 1.0e-9)))
                t0 = tuple(t0.timetuple())
                # if start bound is Dec-31, advance to next year
                t0 = (t0[0] + 1) if (t0[1:3] == (12, 31)) else t0[0]

                t1 = ds["time_bnds"].values[-1][-1] + offset
                t1 = datetime.fromtimestamp(int(np.ceil(int(t1) * 1.0e-9)))
                t1 = tuple(t1.timetuple())
                # if end bound is Jan-1, fall back to previous year
                t1 = (t1[0] - 1) if (t1[1:3] == (1, 1)) else t1[0]

            else:
                # return very obvious incorrect dates to alert the
                # user that the inferred time range failed
                t0 = 9999
                t1 = 9999

        else:
            t0 = tuple(ds["time_bnds"].values[0][0].timetuple())[0]

            # if end bound is Jan-1, fall back to previous year
            t1 = tuple(ds["time_bnds"].values[-1][-1].timetuple())
            t1 = (t1[0] - 1) if (t1[1:3] == (1, 1)) else t1[0]

    else:
        t0 = int(ds["time"].isel({"time": 0}).dt.strftime("%Y"))
        t1 = int(ds["time"].isel({"time": -1}).dt.strftime("%Y"))

    return (t0, t1)


def discover_ts_dir(path, default="ts/monthly"):
    """Find the directory with the longest timeseries chunk

    Parameters
    ----------
    path : str, path-like
        Path to top-level (.../pp) post-processing directory
    default : str, optional
        Time resolution to scan, by default "ts/monthly"

    Returns
    -------
    str, path-like
        Full path to timeseries directory with the longest chunks
    """

    # combine root pp path with the default chunk
    path = f"{path}/{default}"
    path = fixdir(path)

    # get a list of subdirectories
    subdirs = [f.path for f in os.scandir(path) if f.is_dir()]
    subdirs = [dirname.replace(f"{path}/", "") for dirname in subdirs]

    # chomp the last part of the path to obtain the numerical chunk length
    subdirs = [int(dirname.replace("yr", "")) for dirname in subdirs]

    # reconstruct and return the path
    return_path = f"{path}/{max(subdirs)}yr/"
    return fixdir(return_path)


def extract_from_tar(tar, member):
    """Loads Xarray DataSet in memory from a file contained inside a tar file

    Parameters
    ----------
    tar : tarfile.TarFile
        TarFile object generated through tarfile.open('file')
    member : str
        Name of dataset to extract to memory

    Returns
    -------
    xarray.DataSet
        In-memory Xarray dataset
    """

    if member not in tar.getnames():
        member = "./" + member
    f = tar.extractfile(member)
    data = f.read()

    # the line below is retained for NetCDF4 library reference
    # dataset = netCDF4.Dataset("in-mem-file", mode="r", memory=data)

    dataset = xr.open_dataset(data)

    return dataset


def fixdir(path):
    """Ensures a path string does not contain a double-slash

    Parameters
    ----------
    path : str, path-like
        string containing a path

    Returns
    -------
    string with double slashes removed

    """
    return path.replace("//", "/")


def generate_basin_masks(basin_code, basin=None):
    """Returns 2-D array mask (1s/0s) for common pre-defined
    basins and regions.

    Parameters
    ----------
    basin_code : numpy.ndarray
        2-dimensional array of CMIP-convention basin codes
    basin : str or int, optional
        Name of basin to calculate. Options are "atlantic_arctic"
        and "indo_pacific". An integer basin code may also be 
        passed. By default None

    Returns
    -------
    numpy.ndarray
        Basin mask of 1s and 0s.
    """
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


def image_handler(figs, dictArgs, filename="./figure"):
    """ Generic OM4Labs image handler. Depending on the framework mode,
    this handler either saves a matplotlib figure handle to disk or
    returns an in-memory image buffer

    Parameters
    ----------
    figs : matplotlib.Figure or list
        Matplotlib figure handle or list of figure handles
    dictArgs : dict
        Dictionary of parsed command-line options
    filename : str, optional
        Figure filename, by default "./figure"

    Returns
    -------
    io.BytesIO
        In-memory image buffers
    """

    imgbufs = []
    numfigs = len(figs)

    # test if output directory exists
    if dictArgs["outdir"] != "./":
        pathlib.Path(dictArgs["outdir"]).mkdir(parents=True, exist_ok=True)

    if not isinstance(filename, list):
        filename = [filename]

    assert (
        len(filename) == numfigs
    ), "Number of figure handles and file names do not match."

    if dictArgs["interactive"] is True:
        plt.show()

    else:
        for n, fig in enumerate(figs):
            if dictArgs["format"] == "stream":
                imgbuf = io.BytesIO()
                fig.savefig(imgbuf, format="png", bbox_inches="tight")
                imgbufs.append(imgbuf)
            else:
                fig.savefig(
                    f"{filename[n]}.png",
                    format=dictArgs["format"],
                    dpi=150,
                    bbox_inches="tight",
                )

    return imgbufs


def infer_and_assign_coord(ds, da, coordname):
    """ infer what the coord name is and assign it to dataarray """
    assigned_coordname = try_variable_from_list(
        list(ds.variables), possible_names[coordname]
    )
    if (assigned_coordname is not None) and (assigned_coordname in da.dims):
        da = da.rename({assigned_coordname: f"assigned_{coordname}"})
    return da


def read_data(ds, possible_variable_names):
    """ read data from one file """

    # find the appropriate variable names
    varname = try_variable_from_list(list(ds.variables), possible_variable_names)
    if varname is None:
        raise ValueError(f"no suitable variable found in dataset")

    da = ds[varname]
    da = infer_and_assign_coord(ds, da, "lon")
    da = infer_and_assign_coord(ds, da, "lat")
    da = infer_and_assign_coord(ds, da, "time")
    da = infer_and_assign_coord(ds, da, "depth")
    da = infer_and_assign_coord(ds, da, "interfaces")

    return da


def standard_grid_cell_area(lat, lon, rE=6371.0e3):
    """ computes the cell area for a standard spherical grid """

    warnings.warn(
        "standard_grid_cell_area is deprecated, use compute_area_regular_grid",
        DeprecationWarning,
    )

    dLat = lat[1] - lat[0]
    dLon = lon[1] - lon[0]
    area = np.empty((len(lat), len(lon)))
    for j in range(0, len(lat)):
        for i in range(0, len(lon)):
            lon1 = lon[i] + dLon / 2
            lon0 = lon[i] - dLon / 2
            lat1 = lat[j] + dLat / 2
            lat0 = lat[j] - dLat / 2
            area[j, i] = (
                (np.pi / 180.0)
                * rE
                * rE
                * np.abs(np.sin(np.radians(lat0)) - np.sin(np.radians(lat1)))
                * np.abs(lon0 - lon1)
            )
    return area


def subset_data(da, coordname, subset):
    """ subset (float or slice) dataarray along coord """
    if coordname in da.coords:
        da = da.sel({coordname: subset})
    return da


def simple_average(da, coordname):
    """ average """
    if coordname in da.coords:
        da = da.mean(dim=coordname)
    return da


def copy_coordinates(da1, da2, coords):
    """ copy coordinates of da1 into da2 """
    for coord in coords:
        da2[coord] = da1[coord]

    return da2


def compute_area_regular_grid(ds, Rearth=6378e3):
    """ compute the cells area on a regular grid """

    rfac = 2 * np.pi * Rearth / 360

    up = {"bnds": 1}
    down = {"bnds": 0}
    if "time" in ds["lon_bnds"].dims:
        up.update({"time": 0})
        down.update({"time": 0})

    dx1d = rfac * (ds["lon_bnds"].isel(up) - ds["lon_bnds"].isel(down))
    dy1d = rfac * (ds["lat_bnds"].isel(up) - ds["lat_bnds"].isel(down))

    dx2d, dy2d = np.meshgrid(dx1d, dy1d)
    _, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)

    dx = dx2d * np.cos(2 * np.pi * lat2d / 360)
    dy = dy2d
    area = dx * dy
    return area


def is_symmetric(dset, x_center="xh", x_corner="xq", y_center="yh", y_corner="yq"):
    """Determines if ocean model output is on a symmetric grid

    Parameters
    ----------
    dset : [type]
        [description]
    x_center : str, optional
        Name of x-cell centers dimension, by default "yxh"
    x_corner : str, optional
        Name of x-cell corners dimension, by default "xq"
    y_center : str, optional
        Name of y-cell centers dimension, by default "yh"
    y_corner : str, optional
        Name of y-cell corners dimension, by default "yq"

    Returns
    -------
    bool
        True if grid is symmetric
    """

    if (len(dset[x_corner]) == len(dset[x_center])) and (
        len(dset[y_corner]) == len(dset[y_center])
    ):
        out = False
    elif (len(dset[x_corner]) == len(dset[x_center]) + 1) and (
        len(dset[y_corner]) == len(dset[y_center]) + 1
    ):
        out = True
    else:
        raise ValueError("unsupported combination of coordinates")
    return out


def grid_from_supergrid(ds, point_type="t", outputgrid="nonsymetric"):
    """Subsample super grid to obtain geolon, geolat, and cell area

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing variables from the supergrid
    point_type : str, optional
        Requested grid type of t|q|u|v, by default "t"
    outputgrid : str, optional
        Either "symetric" or "nonsymetric", default is "nonsymetric"

    Returns
    -------
    geolat : xarray.DataArray
        2-dimensional Earth-centric latitude coordinates
    geolon : xarray.DataArray
        2-dimensional Earth-centric longitude coordinates
    area : xarray.DataArray
        Array of cell areas with dimension (geolat,geolon)
    """

    geolat = subsample_supergrid(ds, "y", point_type, outputgrid=outputgrid)
    geolon = subsample_supergrid(ds, "x", point_type, outputgrid=outputgrid)
    area = sum_on_supergrid(ds, "area", point_type, outputgrid=outputgrid)

    return geolat, geolon, area


def horizontal_grid(
    dictArgs=None,
    point_type="t",
    coords=None,
    outputgrid="nonsymetric",
    output_type="xarray",
):
    """Returns horizontal grid parameters based on the values of the CLI
    arguments and the presence of intake catalogs.

    The requested grid can either tracer points (t), corner points (q),
    zonal velocity points (u), or meridional velocity points (v).

    The corresponding basin mask for the grid is also returned.

    The nominal x and y 1-D coordinates are provided.  Since these are
    non-physical,however, they should only be used for plotting purposes.

    If `dictArgs` is omitted, the function returns a standard 1x1
    spherical grid.

    Parameters
    ----------
    dictArgs : dict, optional
        dictionary of arguments obtained from the CLI parser, by default None
    point_type : str, optional
        Requested grid type of t|q|u|v, by default "t"
    coords : tuple, optional
        target xarray coordinates
    outputgrid : str, optional
        Either "symetric" or "nonsymetric", default is "nonsymetric"
    output_type : str, optional
        Specify output format of either "xarray" or "numpy", by default "xarray"

    Returns
    -------
    xarray.Dataset or tuple
        Arrays of geolat, geolon, nominal_x, nominal_y, area, and basin
    """
    point_type = point_type.upper()

    # if verbose if present in dictArgs that was generated by the parser,
    # that value takes precedence over the kwarg version
    verbose = dictArgs["verbose"] if "verbose" in dictArgs else False
    basin_file = dictArgs["basin"] if "basin" in dictArgs else None

    if dictArgs is None:
        x = np.arange(0.5, 360.5, 1.0)
        y = np.arange(-89.5, 90.5, 1.0)
        area = standard_grid_cell_area(y, x)
        geolon, geolat = np.meshgrid(x, y)
        geolon = xr.DataArray(geolon, dims=("y", "x"), coords={"y": y, "x": x})
        geolat = xr.DataArray(geolat, dims=("y", "x"), coords={"y": y, "x": x})
        area = xr.DataArray(area, dims=("y", "x"), coords={"y": y, "x": x})
        nominal_x = geolon[geolon.dims[-1]]
        nominal_y = geolat[geolat.dims[-2]]

    elif dictArgs["hgrid"] is not None:
        if verbose:
            print("Using optional hgrid file for horizontal grid.")
        ds = xr.open_dataset(dictArgs["hgrid"])
        geolat, geolon, area = grid_from_supergrid(
            ds, point_type, outputgrid=outputgrid
        )
        wet = infer_wet_mask(dictArgs, coords=coords, point_type=point_type)
        warnings.warn(
            "Inferring wet mask from topography. Consider using ocean_static.nc"
        )

    elif dictArgs["static"] is not None:
        if verbose:
            print("Using optional static file for horizontal grid.")

        ds = xr.open_dataset(dictArgs["static"])

        if point_type == "T":
            geolat = ds["geolat"]
            geolon = ds["geolon"]
            wet = ds["wet"]
            area = ds["areacello"]

        elif point_type == "U":
            geolat = ds["geolat_u"]
            geolon = ds["geolon_u"]
            wet = ds["wet_u"]
            area = ds["areacello_cu"]

        elif point_type == "V":
            geolat = ds["geolat_v"]
            geolon = ds["geolon_v"]
            wet = ds["wet_v"]
            area = ds["areacello_cv"]

        else:
            raise ValueError("Unknown point type. Must be T, U, or V")

    elif dictArgs["gridspec"] is not None:
        if verbose:
            print("Using optional gridspec tar file for horizontal grid.")
        tar = tf.open(dictArgs["gridspec"])
        ds = extract_from_tar(tar, "ocean_hgrid.nc")
        geolat, geolon, area = grid_from_supergrid(
            ds, point_type, outputgrid=outputgrid
        )
        wet = infer_wet_mask(dictArgs, coords=coords, point_type=point_type)
        warnings.warn(
            "Inferring wet mask from topography. Consider using ocean_static.nc"
        )

    elif dictArgs["platform"] is not None and dictArgs["config"] is not None:
        if verbose:
            print(
                f"Using {dictArgs['platform']} {dictArgs['config']} intake catalog for horizontal grid."
            )
        cat = open_intake_catalog(dictArgs["platform"], dictArgs["config"])
        ds = cat["ocean_hgrid"].to_dask()
        geolat, geolon, area = grid_from_supergrid(
            ds, point_type, outputgrid=outputgrid
        )
        wet = infer_wet_mask(dictArgs, coords=coords, point_type=point_type)
        warnings.warn(
            "Inferring wet mask from topography. Consider using ocean_static.nc"
        )

    result = xr.Dataset()
    result["geolat"] = geolat
    result["geolon"] = geolon
    result["area"] = area
    result["wet"] = wet

    # nominal coordinates
    result["nominal_x"] = result.geolon.max(axis=-2)
    result["nominal_y"] = result.geolat.max(axis=-1)

    if result["nominal_y"].min() >= 0:
        warnings.warn("Nominal y latitude is positive definite. May be incorrect.")

    if result["nominal_x"].max() > 360:
        warnings.warn("Nominal x longitude > 360. May be incorrect.")

    # -- process basin codes while we are here
    if basin_file is not None:
        if verbose:
            print("Using optional file for basin code specification.")
        ds = xr.open_dataset(dictArgs["hgrid"])
        result["basin"] = ds.basin
    else:
        result["basin"] = generate_basin_codes(result, lon="geolon", lat="geolat")
    result["basin"] = result.basin.fillna(0.0)

    if output_type == "numpy":
        geolat = np.array(result.geolat.to_masked_array())
        geolon = np.array(result.geolon.to_masked_array())
        nominal_x = np.array(result.nominal_x.to_masked_array())
        nominal_y = np.array(result.nominal_y.to_masked_array())
        area = np.array(result.area.to_masked_array())
        basin = np.array(result.basin.to_masked_array())
        result = (geolat, geolon, nominal_x, nominal_y, area, basin)

    return result


def open_intake_catalog(platform, config):
    """Returns an Intake catalog for a specified platform and config

    Uses the package resources included in the om4labs distribution
    to determine the directory of the intake catalogs, unless it
    is overridden by the "OM4LABS_CATALOG_DIR" environment var.

    Parameters
    ----------
    platform : str
        Site description, e.g. "gfdl", "orion", "testing"
    config : str
        Model configuration, e.g. "OM4p5", "OM4p25"

    Returns
    -------
    intake.catalog.Catalog
        Intake catalog corresponding to specified platform/config
    """

    catalog_str = f"{config}_catalog_{platform}.yml"

    if "OM4LABS_CATALOG_DIR" in os.environ.keys():
        catfile = f"{os.environ['OM4LABS_CATALOG_DIR']}/{catalog_str}"
    else:
        catfile = pkgr.resource_filename("om4labs", f"catalogs/{catalog_str}")

    cat = intake.open_catalog(catfile)

    return cat


def read_topography(dictArgs, coords=None, point_type="t"):
    """Returns topography field based on the values of the CLI
    arguments and the presence of intake catalogs.

    Parameters
    ----------
    dictArgs : dict, optional
        dictionary of arguments obtained from the CLI parser, by default None
    coords : tuple, optional
        target xarray coordinates
    point_type : str, optional
        Requested grid type of t|q|u|v, by default "t"

    Returns
    -------
    xarray.DataArray
        topography array
    """

    point_type = point_type.upper()

    verbose = dictArgs["verbose"] if "verbose" in dictArgs else False

    if dictArgs["topog"] is not None:
        if verbose:
            print("Using optional topg file for depth field")
        ds = xr.open_dataset(dictArgs["topog"])

    elif dictArgs["static"] is not None:
        if verbose:
            print("Using optional static file for depth field.")
        ds = xr.open_dataset(dictArgs["static"])

    elif dictArgs["gridspec"] is not None:
        if verbose:
            print("Using optional gridspec tar file for depth field.")
        tar = tf.open(dictArgs["gridspec"])
        ds = extract_from_tar(tar, "ocean_topog.nc")

    elif dictArgs["platform"] is not None and dictArgs["config"] is not None:
        if verbose:
            print(
                f"Using {dictArgs['platform']} {dictArgs['config']} intake catalog for depth field."
            )
        cat = open_intake_catalog(dictArgs["platform"], dictArgs["config"])
        ds = cat["topog"].to_dask()

    if "deptho" in list(ds.variables):
        depth = ds.deptho
    elif "depth" in list(ds.variables):
        depth = ds.depth

    coords = xr.Dataset(coords)
    xedge = "outer" if (len(coords.xq) == len(coords.xh) + 1) else "right"
    yedge = "outer" if (len(coords.yq) == len(coords.yh) + 1) else "right"
    out_grid = Grid(
        coords,
        coords={"X": {"center": "xh", xedge: "xq"}, "Y": {"center": "yh", yedge: "yq"}},
        periodic=["X"],
    )

    if point_type == "V":
        depth = out_grid.interp(depth, "Y", boundary="fill")
    elif point_type == "U":
        depth = out_grid.interp(depth, "X", boundary="fill")

    return depth


def infer_wet_mask(dictArgs, coords=None, point_type="t"):
    """Infers the model's wet mask based on the model topography

    Parameters
    ----------
    dictArgs : dict, optional
        dictionary of arguments obtained from the CLI parser, by default None
    coords : tuple, optional
        target xarray coordinates
    point_type : str, optional
        Requested grid type of t|q|u|v, by default "t"

    Returns
    -------
    xarray.DataArray
        wet mask of ocean=1, land=0
    """
    depth = read_topography(dictArgs, coords=coords, point_type=point_type)
    depth = xr.where(depth.isnull(), 0.0, depth)
    return xr.where(depth > 0.0, 1.0, 0.0)


def annual_cycle(ds, var):
    """Compute annual cycle climatology"""
    # Make a DataArray with the number of days in each month, size = len(time)
    if hasattr(ds.time, "calendar"):
        cal = ds.time.calendar
    elif hasattr(ds.time, "calendar_type"):
        cal = ds.time.calendar_type.lower()
    else:
        cal = "standard"

    if cal.lower() in ["noleap", "365day"]:
        # always calculate days in month based on year 1 (non-leap year)
        month_length = [calendar.monthrange(1, x.month)[1] for x in ds.time.to_index()]
    else:
        # use real year/month combo to calculate days in month
        month_length = [
            calendar.monthrange(x.year, x.month)[1] for x in ds.time.to_index()
        ]

    month_length = xr.DataArray(month_length, coords=[ds.time], name="month_length")

    # Calculate the weights by grouping by 'time.season'.
    # Conversion to float type ('astype(float)') only necessary for Python 2.x
    weights = (
        month_length.groupby("time.month") / month_length.groupby("time.month").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.month").sum().values, np.ones(12))

    # Calculate the weighted average
    ds_weighted = (ds[var] * weights).groupby("time.month").sum(dim="time")

    return ds_weighted


def add_matrix_NaNs(regridder):
    """Helper function to set masked points to NaN instead of zero"""
    X = regridder.weights
    M = scipy.sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.weights = scipy.sparse.coo_matrix(M)
    return regridder


def curv_to_curv(src, dst, reuse_weights=False):
    regridder = xe.Regridder(src, dst, "bilinear", reuse_weights=reuse_weights)
    if version.parse(xe.__version__) < version.parse("0.4.0"):
        regridder = add_matrix_NaNs(regridder)
    return regridder(src)
