# need a data_read, data_sel, data_reduce

import numpy as np
import argparse
import intake
import io
import signal
import sys
import matplotlib.pyplot as plt
import pkg_resources as pkgr
import tarfile as tf
import xarray as xr

try:
    from om4labs.helpers import try_variable_from_list
except ImportError:
    # DORA mode, works without install.
    # reads from current directory
    from helpers import try_variable_from_list

from static_downsampler.static import sum_on_supergrid
from static_downsampler.static import subsample_supergrid

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


def extract_from_tar(tar, member):
    """
    Function to extract a single netCDF file from within
    an uncompressed tarfile
    """
    if member not in tar.getnames():
        member = "./" + member
    f = tar.extractfile(member)
    data = f.read()
    # the line below is retained for NetCDF4 library reference
    # dataset = netCDF4.Dataset("in-mem-file", mode="r", memory=data)
    dataset = xr.open_dataset(data)
    return dataset


def image_handler(figs, dictArgs, filename="./figure"):
    """Generic routine for image handling"""

    imgbufs = []
    numfigs = len(figs)

    if not isinstance(filename, list):
        filename = [filename]

    assert (
        len(filename) == numfigs
    ), "Number of figure handles and file names do not match."

    if dictArgs["interactive"] is True:
        plt.ion()
        for n, fig in enumerate(figs):
            plt.show(fig)

        def _signal_handler(sig, frame):
            print("Complete!")
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        print("Press ctrl+c to exit...")
        signal.pause()
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


def horizontal_grid(dictArgs, point_type="t"):

    verbose = dictArgs["verbose"]

    geolon = None
    geolat = None
    area = None

    point_type = point_type.upper()

    if dictArgs["hgrid"] is not None:
        if verbose:
            print("Using optional hgrid file for horizontal grid.")
        ds = xr.open_dataset(dictArgs["hgrid"])
        geolat = subsample_supergrid(ds, "y", point_type)
        geolon = subsample_supergrid(ds, "x", point_type)
        area = sum_on_supergrid(ds, "area", point_type)

    elif dictArgs["static"] is not None:
        if verbose:
            print("Using optional static file for horizontal grid.")

        ds = xr.open_dataset(dictArgs["static"])

        if point_type == "T":
            geolat = ds["geolat"]
            geolon = ds["geolon"]
            area = ds["areacello"]

        elif point_type == "U":
            geolat = ds["geolat_u"]
            geolon = ds["geolon_u"]
            area = ds["areacello_cu"]

        elif point_type == "V":
            geolat = ds["geolat_v"]
            geolon = ds["geolon_v"]
            area = ds["areacello_cv"]

        else:
            raise ValueError("Unknown point type. Must be T, U, or V")

    elif dictArgs["gridspec"] is not None:
        if verbose:
            print("Using optional gridspec tar file for horizontal grid.")
        tar = tf.open(dictArgs["gridspec"])
        ds = extract_from_tar(tar, "ocean_hgrid.nc")
        geolat = subsample_supergrid(ds, "y", point_type)
        geolon = subsample_supergrid(ds, "x", point_type)
        area = sum_on_supergrid(ds, "area", point_type)

    elif dictArgs["platform"] is not None and dictArgs["config"] is not None:
        if verbose:
            print(
                f"Using {dictArgs['platform']} {dictArgs['config']} intake catalog for horizontal grid."
            )
        cat = open_intake_catalog(dictArgs["platform"], dictArgs["config"])
        ds = cat["ocean_hgrid"].to_dask()
        geolat = subsample_supergrid(ds, "y", point_type)
        geolon = subsample_supergrid(ds, "x", point_type)
        area = sum_on_supergrid(ds, "area", point_type)

    assert geolon is not None, "Unable to obtain geolon"
    assert geolat is not None, "Unable to obtain geolat"
    assert area is not None, "Unable to obtain area"

    geolat = np.array(geolat.to_masked_array())
    geolon = np.array(geolon.to_masked_array())
    area = np.array(area.to_masked_array())

    return geolat, geolon, area


def open_intake_catalog(platform, config):
    cat_platform = f"catalogs/{config}_catalog_{platform}.yml"
    catfile = pkgr.resource_filename("om4labs", cat_platform)
    cat = intake.open_catalog(catfile)
    return cat
