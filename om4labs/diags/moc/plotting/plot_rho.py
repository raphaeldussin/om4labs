import warnings
import palettable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from om4labs.m6plot.formatting import VerticalSplitScale


def plot_rho_panel(ax, arr, levels=None, title=None, annot=None):
    """Function to individual panel of overturning streamfunction in sigma space

    Parameters
    ----------
    ax : matplotlib.axes.axis
        axis handle to use for plotting
    arr : xarray.DataArray
        array of overturning streamfunction
    levels : list, optional
        contour intervals, by default None
    title : str, optional
        panel title at the top left corner, by default None
    annot : str, optional
        panel annotation at the top right corner, by default None

    Returns
    -------
    matplotlib.collections.QuadMesh
        output from pcolormesh
    """

    # color shading
    cb = arr.plot(ax=ax, add_colorbar=False)

    # contour levels
    if levels is None:
        levels = np.arange(-40, 45, 5)

    # draw contours
    arr.plot.contour(ax=ax, levels=levels, colors=["k"], linewidths=0.5)

    # formatting vertical axis
    ax.set_yscale("splitscale", zval=[1037.2, 1036.5, 1028])

    # set labels
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("sigma2")
    ax.text(0.01, 1.02, title, ha="left", fontsize=10, transform=ax.transAxes)
    ax.text(0.98, 1.02, annot, ha="right", fontsize=10, transform=ax.transAxes)

    return cb


def plot_rho(dset, otsfn, label=None):
    """MOC plotting script for rho2 overturning

    Parameters
    ----------
    dset : xarray.Dataset
        Dataset containing grid and mask fields
    otsfn : xarray.DataArray
        DataArray containing the overturning streamfunction
    label : str, optional
        Text label to add to plot, by default None

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle
    """

    # setup figure handle
    fig = plt.figure(figsize=(8.5, 11))

    # format date string
    dates = dset.attrs["dates"]
    dates = f"Years {dates[0]} - {dates[1]}"

    # Panel 1 (top): Atlantic-Arctic
    ax1 = plt.subplot(3, 1, 1)
    levels = np.arange(-42, 45, 3)
    title = "a. Atlantic-Arctic MOC [Sv]"
    cb1 = plot_rho_panel(
        ax1, otsfn.isel(basin=0), levels=levels, title=title, annot=dates
    )

    # Panel 2 (middle): Indo-Pacific
    ax2 = plt.subplot(3, 1, 2)
    title = "b. Indo-Pacific MOC [Sv]"
    cb2 = plot_rho_panel(ax2, otsfn.isel(basin=1), levels=levels, title=title)

    # Panel 3 (bottom): Global
    ax3 = plt.subplot(3, 1, 3)
    title = "c. Global MOC [Sv]"
    cb3 = plot_rho_panel(ax3, otsfn.isel(basin=2), levels=levels, title=title)

    plt.suptitle(label)

    return fig
