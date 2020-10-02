# need a data_read, data_sel, data_reduce

import numpy as np
import argparse
import io
import signal
import sys
import matplotlib.pyplot as plt

try:
    from om4labs.helpers import try_variable_from_list
except ImportError:
    # DORA mode, works without install.
    # reads from current directory
    from helpers import try_variable_from_list


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
