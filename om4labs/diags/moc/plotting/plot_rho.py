import warnings
import palettable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from om4labs.m6plot.formatting import VerticalSplitScale


def plot_rho_panel(ax, arr, lat, rho, levels=None, title=None):
    """Function to individual panel of overturning streamfunction in sigma space

    Parameters
    ----------
    ax : matplotlib.axes.axis
        axis handle to use for plotting
    arr : numpy.ma.MaskedArray
        Input array with dimensions (basin,rho,lat)
    lat : numpy.array
        1-D array of latitude values
    rho : numpy.array
        1-D array of rho values
    levels : list, optional
        contour intervals, by default None
    title : str, optional
        panel title at the top left corner, by default None

    Returns
    -------
    matplotlib.collections.QuadMesh
        output from pcolormesh
    """

    # contour levels
    if levels is None:
        levels = np.arange(-40, 45, 5)

    cb = ax.contourf(lat, rho, arr, levels=levels, cmap="RdBu_r")
    cs = ax.contour(lat, rho, arr, levels=levels, colors=["k"], linewidths=0.5)

    # formatting vertical axis
    ax.set_yscale("splitscale", zval=[1037.2, 1036.5, 1028])

    # set labels
    ax.set_ylabel("sigma2")
    ax.text(0.01, 1.02, title, ha="left", fontsize=10, transform=ax.transAxes)

    return cb


def plot_rho(otsfn, lat, rho, label=None, dates=None):
    """

    Parameters
    ----------
    otsfn : numpy.ma.MaskedArray
        Input array with dimensions (basin,rho,lat)
    rho : numpy.array
        1-D array of rho values
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

    # setup figure handle
    fig = plt.figure(figsize=(8.5, 11))

    # Panel 1 (top): Atlantic-Arctic
    ax1 = plt.subplot(3, 1, 1)
    levels = np.arange(-42, 45, 3)
    title = "a. Atlantic-Arctic MOC [Sv]"
    cb1 = plot_rho_panel(ax1, otsfn[0], lat, rho, levels=levels, title=title)

    # Panel 2 (middle): Indo-Pacific
    ax2 = plt.subplot(3, 1, 2)
    title = "b. Indo-Pacific MOC [Sv]"
    cb2 = plot_rho_panel(ax2, otsfn[1], lat, rho, levels=levels, title=title)

    # Panel 3 (bottom): Global
    ax3 = plt.subplot(3, 1, 3)
    title = "c. Global MOC [Sv]"
    cb3 = plot_rho_panel(ax3, otsfn[2], lat, rho, levels=levels, title=title)

    plt.suptitle(label)
    # format date string
    if dates is not None:
        dates = f"Years {dates[0]} - {dates[1]}"
        ax1.text(0.98, 1.02, dates, ha="right", transform=ax1.transAxes)

    return fig
