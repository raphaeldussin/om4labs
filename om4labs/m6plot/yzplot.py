from . import cm
from . import coords
from . import formatting
from . import stats
import matplotlib.pyplot as plt
import numpy as np


def yzplot(
    field,
    y=None,
    z=None,
    ylabel=None,
    yunits=None,
    zlabel=None,
    zunits=None,
    splitscale=None,
    title="",
    suptitle="",
    clim=None,
    colormap=None,
    extend=None,
    centerlabels=False,
    nbins=None,
    landcolor=[0.5, 0.5, 0.5],
    aspect=[16, 9],
    resolution=576,
    axis=None,
    ignore=None,
    save=None,
    debug=False,
    show=False,
    interactive=False,
):
    """
  Renders section plot of scalar field, field(x,z).

  Arguments:
  field       Scalar 2D array to be plotted.
  y           y (or x) coordinate (1D array). If y is the same size as field then x is treated as
              the cell center coordinates.
  z           z coordinate (1D or 2D array). If z is the same size as field then y is treated as
              the cell center coordinates.
  ylabel      The label for the x axis. Default 'Latitude'.
  yunits      The units for the x axis. Default 'degrees N'.
  zlabel      The label for the z axis. Default 'Elevation'.
  zunits      The units for the z axis. Default 'm'.
  splitscale    A list of depths to define equal regions of projection in the vertical, e.g. [0.,-1000,-6500]
  title       The title to place at the top of the panel. Default ''.
  suptitle    The super-title to place at the top of the figure. Default ''.
  clim        A tuple of (min,max) color range OR a list of contour levels. Default None.
  colormap    The name of the colormap to use. Default None.
  extend      Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels If True, will move the colorbar labels to the middle of the interval. Default False.
  nbins       The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor   An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  aspect      The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution  The vertical resolution of the figure given in pixels. Default 720.
  axis         The axis handle to plot to. Default None.
  ignore      A value to use as no-data (NaN). Default None.
  save        Name of file to save figure in. Default None.
  debug       If true, report stuff for debugging. Default False.
  show        If true, causes the figure to appear on screen. Used for testing. Default False.
  interactive If true, adds interactive features such as zoom, close and cursor. Default False.
  """

    c = cm.dunne_pm()
    c = cm.dunne_rainbow()

    # Create coordinates if not provided
    ylabel, yunits, zlabel, zunits = formatting.createYZlabels(
        y, z, ylabel, yunits, zlabel, zunits
    )
    if debug:
        print("y,z label/units=", ylabel, yunits, zlabel, zunits)
    if len(y) == z.shape[-1]:
        y = coords.expand(y)
    elif len(y) == z.shape[-1] + 1:
        y = y
    else:
        raise Exception(
            "Length of y coordinate should be equal or 1 longer than horizontal length of z"
        )
    if ignore is not None:
        maskedField = np.ma.masked_array(field, mask=[field == ignore])
    else:
        maskedField = field.copy()
    yCoord, zCoord, field2 = coords.section2quadmesh(y, z, maskedField)

    # Diagnose statistics
    sMin, sMax, sMean, sStd, sRMS = stats.calc(
        maskedField, stats.yzWeight(y, z), debug=debug
    )
    yLims = np.amin(yCoord), np.amax(yCoord)
    zLims = coords.boundaryStats(zCoord)

    # Choose colormap
    if nbins is None and (clim is None or len(clim) == 2):
        nbins = 35
    if colormap is None:
        colormap = cm.chooseColorMap(sMin, sMax)
    cmap, norm, extend = cm.chooseColorLevels(
        sMin, sMax, colormap, clim=clim, nbins=nbins, extend=extend
    )

    if axis is None:
        formatting.setFigureSize(aspect, resolution, debug=debug)
        # plt.gcf().subplots_adjust(left=.10, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
        axis = plt.gca()
    plt.pcolormesh(yCoord, zCoord, field2, cmap=cmap, norm=norm)
    if interactive:
        addStatusBar(yCoord, zCoord, field2)
    cb = plt.colorbar(fraction=0.08, pad=0.02, extend=extend)
    if centerlabels and len(clim) > 2:
        cb.set_ticks(0.5 * (clim[:-1] + clim[1:]))
    axis.set_facecolor(landcolor)
    if splitscale is not None:
        for zzz in splitscale[1:-1]:
            plt.axhline(zzz, color="k", linestyle="--")
        axis.set_yscale("splitscale", zval=splitscale)
    plt.xlim(yLims)
    plt.ylim(zLims)
    axis.annotate(
        "max=%.5g\nmin=%.5g" % (sMax, sMin),
        xy=(0.0, 1.01),
        xycoords="axes fraction",
        verticalalignment="bottom",
        fontsize=10,
    )
    if sMean is not None:
        axis.annotate(
            "mean=%.5g\nrms=%.5g" % (sMean, sRMS),
            xy=(1.0, 1.01),
            xycoords="axes fraction",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=10,
        )
        axis.annotate(
            " sd=%.5g\n" % (sStd),
            xy=(1.0, 1.01),
            xycoords="axes fraction",
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=10,
        )
    if len(ylabel + yunits) > 0:
        plt.xlabel(formatting.label(ylabel, yunits))
    if len(zlabel + zunits) > 0:
        plt.ylabel(formatting.label(zlabel, zunits))
    if len(title) > 0:
        plt.title(title)
    if len(suptitle) > 0:
        plt.suptitle(suptitle)

    if save is not None:
        plt.savefig(save)
    if interactive:
        addInteractiveCallbacks()
    if show:
        plt.show(block=False)
