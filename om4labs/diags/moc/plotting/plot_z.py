import warnings
import palettable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from om4labs import m6plot

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


def plot_z(otsfn, lat, depth, label=None, dates=None):
    """Function to plot 3-panel MOC figure

    Parameters
    ----------
    otsfn : numpy.ma.MaskedArray
        Input array with dimensions (basin,depth,lat)
    lat : numpy.array
        1-D array of latitude values
    depth : numpy.array
        1-D array of depth values
    label : str, optional
        Experiment name / top line label, by default None
    dates : tuple, optional
        Tuple of date ranges, by default None

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle
    """

    # Set up the figure
    fig = plt.figure(figsize=(8.5, 11))

    # Top panel: Atlantic-Arctic
    arr = otsfn[0]
    ax1 = plt.subplot(3, 1, 1, facecolor="#bbbbbb")
    plot_z_panel(ax1, arr, lat, depth, xlim=(-33, 90))
    ax1.text(
        0.01,
        1.02,
        "a. Atlantic-Arctic",
        ha="left",
        fontsize=12,
        transform=ax1.transAxes,
    )

    # Basin maximum
    basinmax = z_extremes(arr, lat, depth, ylim=(-33, 90), zlim=(500, 2000))
    maxval = str(round(basinmax[2], 1))
    ax1.plot(*basinmax[0:2], marker="o", markersize=5, color="w")
    ax1.text(*basinmax[0:2], maxval, fontsize=16, ha="right", va="top")
    ax1.plot(60, 4500, marker="o", markersize=5, color="w")
    ax1.text(63, 4500, "N. Atl. Max", fontsize=10, ha="left", va="center")

    # Maximum at latitude of RAPID array
    rapid = z_extremes(arr, lat, depth, ylim=(25, 28.0), zlim=(500, 2000))
    maxval = str(round(rapid[2], 1))
    ax1.plot(*rapid[0:2], marker="o", markersize=5, color="#ffff00")
    ax1.text(*rapid[0:2], maxval, fontsize=16, ha="left", va="bottom")
    ax1.plot(60, 5500, marker="o", markersize=5, color="#ffff00")
    ax1.text(63, 5500, "RAPID", fontsize=10, ha="left", va="center")

    # Middle panel: Indo-Pacific
    arr = otsfn[1]
    ax2 = plt.subplot(3, 1, 2, facecolor="#bbbbbb")
    plot_z_panel(ax2, arr, lat, depth, xlim=(-40, 65))
    ax2.text(
        0.01, 1.02, "b. Indo-Pacific", ha="left", fontsize=12, transform=ax2.transAxes
    )

    # Bottom panel: Global
    arr = otsfn[2]
    ax3 = plt.subplot(3, 1, 3, facecolor="#bbbbbb")
    plot_z_panel(ax3, arr, lat, depth)
    ax3.text(0.01, 1.02, "c. Global", ha="left", fontsize=12, transform=ax3.transAxes)

    # Add dates to top panel
    if dates is not None:
        dates = f"Years {dates[0]} - {dates[1]}"
        ax1.text(0.98, 1.02, dates, ha="right", fontsize=12, transform=ax1.transAxes)

    # adjust panel spacing
    plt.subplots_adjust(hspace=0.25)

    # add top label
    plt.suptitle(label)

    return fig


def plot_z_panel(ax, arr, lat, depth, levels=None, xlim=None):
    """Plots an indiviual MOC panel

    Parameters
    ----------
    ax : matplotlib.axes.axis
        Axis handle
    arr : numpy.ma.MaskedArray
        2-D masked array with dimensions (depth,lat)
    lat : numpy.array
        1-D array of latitude values
    depth : numpy.array
        1-D array of depth values
    levels : list, optional
        list of contour intervals, by default None
    xlim : tuple, optional
        latitude range for plotting, by default None
    """

    levels = np.arange(-45, 48, 3) if levels is None else levels
    ax.contourf(lat, depth, arr, levels=levels, cmap="RdBu_r")
    ax.contour(lat, depth, arr, levels=levels, colors=["k"], linewidths=0.5)
    ax.contour(lat, depth, arr, levels=[0.0], colors=["k"], linewidths=1)
    ax.set_yscale("splitscale", zval=[6500.0, 2000.0, 0.0])
    ax.set_xlim(xlim)


def z_extremes(arr, lat, depth, zlim=None, ylim=None):
    """Locate max value of a lat-depth array

    Parameters
    ----------
    arr : numpy.ma.MaskedArray
        2-D masked array with dimensions (depth,lat)
    lat : numpy.array
        1-D array of latitude values
    depth : numpy.array
        1-D array of depth values
    zlim : tuple, optional
        depth range, by default None
    ylim : tuple, optional
        latitude range, by default None

    Returns
    -------
    tuple
        (lat,depth,max. value)
    """

    # broadcast lat and depth to 2D arrays
    lat = np.tile(lat[None, :], (arr.shape[0], 1))
    depth = np.tile(depth[:, None], (1, arr.shape[1]))

    # latitude range mask
    if ylim is not None:
        assert isinstance(ylim, tuple), "ylim must be a tuple"
        ymask = np.where(lat >= ylim[0], 1.0, 0.0)
        ymask = np.where(lat <= ylim[1], ymask, 0.0)
    else:
        ymask = 1.0

    # depth range mask
    if zlim is not None:
        assert isinstance(zlim, tuple), "zlim must be a tuple"
        zmask = np.where(depth >= zlim[0], 1.0, 0.0)
        zmask = np.where(depth <= zlim[1], zmask, 0.0)
    else:
        zmask = 1.0

    # combine masks
    mask = np.array(ymask * zmask)

    # turn mask into a masked array
    if len(mask.shape) > 0:
        mask = np.ma.masked_where(mask == 0.0, mask)

    # multiply array by mask
    arr = arr * mask

    # find indicies of max values
    ind = np.unravel_index(np.ma.argmax(arr, axis=None), arr.shape)

    return (lat[ind], depth[ind], arr[ind])
