import warnings
import palettable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from om4labs import m6plot

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


def annotate_z_extrema(
    ax, y, z, psi, min_lat=-90.0, max_lat=90.0, min_depth=0.0, mult=1.0
):
    """Function to annotate min/max values on z-level MOC plot

    Parameters
    ----------
    ax : matplotlib.axes.axis
         Axis containing plot
    y : numpy.ndarray
        y-coordinate
    z : numpy.ndarray
        z-coordinate
    psi : numpy.ndarray
        Array of overturning streamfuctuon
    min_lat : float, optional
        starting search latitude, by default -90.0
    max_lat : float, optional
        ending search latitude, by default 90.0
    min_depth : float, optional
        minimum search depth, by default 0.0
    mult : float, optional
        scaling factor, by default 1.0
    """
    psiMax = mult * np.amax(
        mult * np.ma.array(psi)[(y >= min_lat) & (y <= max_lat) & (z < -min_depth)]
    )
    idx = np.argmin(np.abs(psi - psiMax))
    (j, i) = np.unravel_index(idx, psi.shape)
    ax.plot(y[j, i], z[j, i], "kx")
    ax.text(y[j, i], z[j, i], "%.1f" % (psi[j, i]))


def create_z_topomask(depth, yh, mask=None):
    """Creates a bottom topography mask for plots

    Parameters
    ----------
    depth : numpy.ndarray
        3-dimensional depth field (zmod)
    yh : np.ndarray
        nominal latitude array
    mask : numpy.ndarray, optional
        optional secondary mask, by default None

    Returns
    -------
    numpy.ma.maskedArray
        topography mask array
    """
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


def plot_z(dset, otsfn, label=None):
    """Plotting script"""

    # get y-coord from geolat
    y = dset.geolat.values
    z = dset.zmod.values
    yh = dset.yh.values
    depth = dset.depth.values
    atlantic_arctic_mask = dset.basin_masks.isel(basin=0)
    indo_pacific_mask = dset.basin_masks.isel(basin=1)
    dates = dset.dates

    if len(z.shape) != 1:
        z = z.min(axis=-1)
    yy = y[:, :].max(axis=-1) + 0 * z

    psi = otsfn.to_masked_array()

    atlantic_topomask, zz = create_z_topomask(depth, yh, atlantic_arctic_mask)
    indo_pacific_topomask, zz = create_z_topomask(depth, yh, indo_pacific_mask)
    global_topomask, zz = create_z_topomask(depth, yh)

    ci = m6plot.formatting.pmCI(0.0, 43.0, 3.0)
    cmap = palettable.cmocean.diverging.Balance_20.get_mpl_colormap()

    fig = plt.figure(figsize=(8.5, 11))
    ax1 = plt.subplot(3, 1, 1, facecolor="gray")
    psiPlot = psi[0, 0]
    plot_z_panel(
        ax1,
        yy,
        z,
        psiPlot,
        ci,
        "a. Atlantic MOC [Sv]",
        cmap=cmap,
        xlim=(-40, 90),
        topomask=atlantic_topomask,
        yh=yh,
        zz=zz,
        dates=dates,
    )
    annotate_z_extrema(ax1, yy, z, psiPlot, min_lat=26.5, max_lat=27.0)
    annotate_z_extrema(ax1, yy, z, psiPlot, max_lat=-33.0)
    annotate_z_extrema(ax1, yy, z, psiPlot)

    ax2 = plt.subplot(3, 1, 2, facecolor="gray")
    psiPlot = psi[0, 1]
    plot_z_panel(
        ax2,
        yy,
        z,
        psiPlot,
        ci,
        "b. Indo-Pacific MOC [Sv]",
        cmap=cmap,
        xlim=(-40, 65),
        topomask=indo_pacific_topomask,
        yh=yh,
        zz=zz,
    )
    annotate_z_extrema(ax2, yy, z, psiPlot, min_depth=2000.0, mult=-1.0)
    annotate_z_extrema(ax2, yy, z, psiPlot)

    ax3 = plt.subplot(3, 1, 3, facecolor="gray")
    psiPlot = psi[0, 2]
    plot_z_panel(
        ax3,
        yy,
        z,
        psiPlot,
        ci,
        "c. Global MOC [Sv]",
        cmap=cmap,
        topomask=global_topomask,
        yh=yh,
        zz=zz,
    )
    annotate_z_extrema(ax3, yy, z, psiPlot, max_lat=-30.0)
    annotate_z_extrema(ax3, yy, z, psiPlot, min_lat=25.0)
    annotate_z_extrema(ax3, yy, z, psiPlot, min_depth=2000.0, mult=-1.0)
    plt.xlabel(r"Latitude [$\degree$N]")

    plt.subplots_adjust(hspace=0.2)

    if label is not None:
        plt.suptitle(label)

    return fig


def plot_z_panel(
    ax,
    y,
    z,
    psi,
    ci,
    title,
    cmap=None,
    xlim=None,
    topomask=None,
    yh=None,
    zz=None,
    dates=None,
):
    """Plotting routine for an individual z-coordinate panel

    Parameters
    ----------
    ax : matplotlib.axes.axis
        axis handle to use for plotting
    y : numpy.ndarray
        y-coordinate to use for contour plots
    z : numpy.ndarray
        z-coordinate to use for contour plots
    psi : numpy.ndarray
        array of overturning streamfunction in z-levels
    ci : list of floats
        list of countour intervals (levels)
    title : str
        title of plot/experiment
    cmap : matplotlib.cm.colormap, optional
        colormap, by default None
    xlim : tuple, optional
        range of latitudes to plot along the x-axis, by default None
    topomask : numpy.ndarray, optional
        topography mask, by default None
    yh : numpy.ndarray, optional
        nominal latitude values for plotting, by default None
    zz : numpy.ndarray, optional
        secondary z-level axis on cell centers for topomask, by default None
    dates : tuple, optional
        date range of dataset, by default None
    """

    if topomask is not None:
        psi = np.array(np.where(psi.mask, 0.0, psi))
    else:
        psi = np.array(np.where(psi.mask, np.nan, psi))

    cs = ax.contourf(y, z, psi, levels=ci, cmap=cmap, extend="both")
    ax.contour(y, z, psi, levels=ci, colors="k", linewidths=0.4)
    ax.contour(y, z, psi, levels=[0], colors="k", linewidths=0.8)

    # shade topography
    if topomask is not None:
        cMap = mpl.colors.ListedColormap(["gray"])
        ax.pcolormesh(yh, -1.0 * zz, topomask, cmap=cMap, shading="auto")

    # set latitude limits
    if xlim is not None:
        ax.set_xlim(xlim)

    # set vertical split scale
    ax.set_yscale("splitscale", zval=[0, -2000, -6500])
    ax.invert_yaxis()

    # add colorbar
    cbar = plt.colorbar(cs)
    cbar.set_label("[Sv]")

    # add labels
    ax.text(0.02, 1.02, title, ha="left", fontsize=10, transform=ax.transAxes)
    plt.ylabel("Elevation [m]")
    if dates is not None:
        assert isinstance(dates, tuple), "Year range should be provided as a tuple."
        datestring = f"Years {dates[0]} - {dates[1]}"
        ax.text(0.98, 1.02, datestring, ha="right", fontsize=10, transform=ax.transAxes)
