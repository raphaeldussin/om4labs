import matplotlib.pyplot as plt
import numpy
import numpy.matlib
from . import cm
from . import coords
from . import formatting
from . import stats


def xyplot(
    field,
    x=None,
    y=None,
    area=None,
    xlabel=None,
    xunits=None,
    ylabel=None,
    yunits=None,
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
    sigma=2.0,
    ignore=None,
    save=None,
    debug=False,
    show=False,
    interactive=False,
    logscale=False,
):
    """
  Renders plot of scalar field, field(x,y).

  Arguments:
  field        Scalar 2D array to be plotted.
  x            x coordinate (1D or 2D array). If x is the same size as field then x is treated as
               the cell center coordinates.
  y            y coordinate (1D or 2D array). If x is the same size as field then y is treated as
               the cell center coordinates.
  area         2D array of cell areas (used for statistics). Default None.
  xlabel       The label for the x axis. Default 'Longitude'.
  xunits       The units for the x axis. Default 'degrees E'.
  ylabel       The label for the y axis. Default 'Latitude'.
  yunits       The units for the y axis. Default 'degrees N'.
  title        The title to place at the top of the panel. Default ''.
  suptitle     The super-title to place at the top of the figure. Default ''.
  clim         A tuple of (min,max) color range OR a list of contour levels. Default None.
  sigma         Sigma range for difference plot autocolor levels. Default is to span a 2. sigma range
  colormap     The name of the colormap to use. Default None.
  extend       Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels If True, will move the colorbar labels to the middle of the interval. Default False.
  nbins        The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor    An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  aspect       The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution   The vertical resolution of the figure given in pixels. Default 720.
  axis         The axis handle to plot to. Default None.
  ignore       A value to use as no-data (NaN). Default None.
  save         Name of file to save figure in. Default None.
  debug        If true, report stuff for debugging. Default False.
  show         If true, causes the figure to appear on screen. Used for testing. Default False.
  interactive  If true, adds interactive features such as zoom, close and cursor. Default False.
  logscale     If true, use logaritmic coloring scheme. Default False.
  """

    c = cm.dunne_pm()
    c = cm.dunne_rainbow()

    # Create coordinates if not provided
    xlabel, xunits, ylabel, yunits = formatting.createXYlabels(
        x, y, xlabel, xunits, ylabel, yunits
    )
    if debug:
        print("x,y label/units=", xlabel, xunits, ylabel, yunits)
    xCoord, yCoord = formatting.createXYcoords(field, x, y)

    # Diagnose statistics
    if ignore is not None:
        maskedField = numpy.ma.masked_array(field, mask=[field == ignore])
    else:
        maskedField = field.copy()
    sMin, sMax, sMean, sStd, sRMS = stats.calc(maskedField, area, debug=debug)
    xLims = coords.boundaryStats(xCoord)
    yLims = coords.boundaryStats(yCoord)

    # Choose colormap
    if nbins is None and (clim is None or len(clim) == 2):
        nbins = 35
    if colormap is None:
        colormap = chooseColorMap(sMin, sMax)
    if clim is None and sStd > 0:
        cmap, norm, extend = cm.chooseColorLevels(
            sMean - sigma * sStd,
            sMean + sigma * sStd,
            colormap,
            clim=clim,
            nbins=nbins,
            extend=extend,
            logscale=logscale,
        )
    else:
        cmap, norm, extend = cm.chooseColorLevels(
            sMin,
            sMax,
            colormap,
            clim=clim,
            nbins=nbins,
            extend=extend,
            logscale=logscale,
        )

    if axis is None:
        formatting.setFigureSize(aspect, resolution, debug=debug)
        # plt.gcf().subplots_adjust(left=.08, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
        axis = plt.gca()
    plt.pcolormesh(xCoord, yCoord, maskedField, cmap=cmap, norm=norm)
    if interactive:
        addStatusBar(xCoord, yCoord, maskedField)
    cb = plt.colorbar(fraction=0.08, pad=0.02, extend=extend)
    if centerlabels and len(clim) > 2:
        cb.set_ticks(0.5 * (clim[:-1] + clim[1:]))
    elif clim is not None and len(clim) > 2:
        cb.set_ticks(clim)
    axis.set_facecolor(landcolor)
    plt.xlim(xLims)
    plt.ylim(yLims)
    axis.annotate(
        "max=%.5g\nmin=%.5g" % (sMax, sMin),
        xy=(0.0, 1.01),
        xycoords="axes fraction",
        verticalalignment="bottom",
        fontsize=10,
    )
    if area is not None:
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
    if len(xlabel + xunits) > 0:
        plt.xlabel(formatting.label(xlabel, xunits))
    if len(ylabel + yunits) > 0:
        plt.ylabel(formatting.label(ylabel, yunits))
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
