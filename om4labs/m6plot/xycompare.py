import matplotlib.pyplot as plt
import numpy
import numpy.matlib
from . import cm
from . import coords
from . import formatting
from . import geoplot
from . import stats
from . import addStatusBar
from . import addInteractiveCallbacks


def xycompare(
    field1,
    field2,
    x=None,
    y=None,
    area=None,
    xlabel=None,
    xunits=None,
    ylabel=None,
    yunits=None,
    title1="",
    title2="",
    title3="A - B",
    addplabel=True,
    suptitle="",
    clim=None,
    colormap=None,
    extend=None,
    centerlabels=False,
    dlim=None,
    dcolormap=None,
    dextend=None,
    centerdlabels=False,
    nbins=None,
    landcolor=[0.5, 0.5, 0.5],
    sector=None,
    webversion=False,
    aspect=None,
    resolution=None,
    axis=None,
    npanels=3,
    sigma=2.0,
    ignore=None,
    save=None,
    debug=False,
    show=False,
    interactive=False,
):
    """
  Renders n-panel plot of two scalar fields, field1(x,y) and field2(x,y).

  Arguments:
  field1        Scalar 2D array to be plotted and compared to field2.
  field2        Scalar 2D array to be plotted and compared to field1.
  x             x coordinate (1D or 2D array). If x is the same size as field then x is treated as
                the cell center coordinates.
  y             y coordinate (1D or 2D array). If x is the same size as field then y is treated as
                the cell center coordinates.
  area          2D array of cell areas (used for statistics). Default None.
  xlabel        The label for the x axis. Default 'Longitude'.
  xunits        The units for the x axis. Default 'degrees E'.
  ylabel        The label for the y axis. Default 'Latitude'.
  yunits        The units for the y axis. Default 'degrees N'.
  title1        The title to place at the top of panel 1. Default ''.
  title2        The title to place at the top of panel 1. Default ''.
  title3        The title to place at the top of panel 1. Default 'A-B'.
  addplabel     Adds a 'A:' or 'B:' to the title1 and title2. Default True.
  suptitle      The super-title to place at the top of the figure. Default ''.
  sector        Restrcit plot to a specific sector. Default 'None' (i.e. global).
  clim          A tuple of (min,max) color range OR a list of contour levels for the field plots. Default None.
  sigma         Sigma range for difference plot autocolor levels. Default is to span a 2. sigma range
  colormap      The name of the colormap to use for the field plots. Default None.
  extend        Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels  If True, will move the colorbar labels to the middle of the interval. Default False.
  dlim          A tuple of (min,max) color range OR a list of contour levels for the difference plot. Default None.
  dcolormap     The name of the colormap to use for the difference plot. Default None.
  dextend       For the difference colorbar. Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerdlabels If True, will move the difference colorbar labels to the middle of the interval. Default False.
  nbins         The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor     An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  aspect        The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution    The vertical resolution of the figure given in pixels. Default 1280.
  axis          The axis handle to plot to. Default None.
  npanels       Number of panels to display (1, 2 or 3). Default 3.
  ignore        A value to use as no-data (NaN). Default None.
  save          Name of file to save figure in. Default None.
  debug         If true, report stuff for debugging. Default False.
  show          If true, causes the figure to appear on screen. Used for testing. Default False.
  webversion    If true, set options specific for displaying figures in a web browser. Default False.
  interactive   If true, adds interactive features such as zoom, close and cursor. Default False.
  """

    # set visual backend
    if interactive is False:
        plt.switch_backend("Agg")
    else:
        plt.switch_backend("qt5agg")

    c = cm.dunne_pm()
    c = cm.dunne_rainbow()

    if (field1.shape) != (field2.shape):
        raise Exception("field1 and field2 must be the same shape")

    # Create coordinates if not provided
    xlabel, xunits, ylabel, yunits = formatting.createXYlabels(
        x, y, xlabel, xunits, ylabel, yunits
    )
    if debug:
        print("x,y label/units=", xlabel, xunits, ylabel, yunits)
    xCoord, yCoord = formatting.createXYcoords(field1, x, y)

    # Establish ranges for sectors
    lonRange, latRange, hspace, titleOffset = geoplot.sectorRanges(sector=sector)

    # Diagnose statistics
    if sector == None or sector == "global":
        if ignore is not None:
            maskedField1 = numpy.ma.masked_array(field1, mask=[field1 == ignore])
        else:
            maskedField1 = field1.copy()
        if ignore is not None:
            maskedField2 = numpy.ma.masked_array(field2, mask=[field2 == ignore])
        else:
            maskedField2 = field2.copy()
    else:
        maskedField1 = regionalMasking(field1, yCoord, xCoord, latRange, lonRange)
        maskedField2 = regionalMasking(field2, yCoord, xCoord, latRange, lonRange)
        areaCopy = numpy.ma.array(area, mask=maskedField1.mask, copy=True)
    s1Min, s1Max, s1Mean, s1Std, s1RMS = stats.calc(maskedField1, area, debug=debug)
    s2Min, s2Max, s2Mean, s2Std, s2RMS = stats.calc(maskedField2, area, debug=debug)
    dMin, dMax, dMean, dStd, dRMS = stats.calc(
        maskedField1 - maskedField2, area, debug=debug
    )
    if s1Mean is not None:
        dRxy = stats.corr(maskedField1 - s1Mean, maskedField2 - s2Mean, area)
    else:
        dRxy = None
    s12Min = min(s1Min, s2Min)
    s12Max = max(s1Max, s2Max)
    xLims = formatting.boundaryStats(xCoord)
    yLims = formatting.boundaryStats(yCoord)
    if debug:
        print("s1: min, max, mean =", s1Min, s1Max, s1Mean)
        print("s2: min, max, mean =", s2Min, s2Max, s2Mean)
        print("s12: min, max =", s12Min, s12Max)

    # Choose colormap
    if nbins is None and (clim is None or len(clim) == 2):
        cBins = 35
    else:
        cBins = nbins
    if nbins is None and (dlim is None or len(dlim) == 2):
        nbins = 35
    if colormap is None:
        colormap = chooseColorMap(s12Min, s12Max)
    cmap, norm, extend = cm.chooseColorLevels(
        s12Min, s12Max, colormap, clim=clim, nbins=cBins, extend=extend
    )

    def annotateStats(axis, sMin, sMax, sMean, sStd, sRMS, webversion=False):
        if webversion == True:
            fontsize = 9
        else:
            fontsize = 10
        axis.annotate(
            "max=%.5g\nmin=%.5g" % (sMax, sMin),
            xy=(0.0, 1.025),
            xycoords="axes fraction",
            verticalalignment="bottom",
            fontsize=fontsize,
        )
        if sMean is not None:
            axis.annotate(
                "mean=%.5g\nrms=%.5g" % (sMean, sRMS),
                xy=(1.0, 1.025),
                xycoords="axes fraction",
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=fontsize,
            )
            axis.annotate(
                " sd=%.5g\n" % (sStd),
                xy=(1.0, 1.025),
                xycoords="axes fraction",
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=fontsize,
            )

    if addplabel:
        preTitleA = "A: "
        preTitleB = "B: "
    else:
        preTitleA = ""
        preTitleB = ""

    if axis is None:
        formatting.setFigureSize(aspect, resolution, npanels=npanels, debug=debug)

    if npanels in [2, 3]:
        axis = plt.subplot(npanels, 1, 1)
        if sector == None or sector == "global":
            plt.pcolormesh(xCoord, yCoord, maskedField1, cmap=cmap, norm=norm)
            if interactive:
                addStatusBar(xCoord, yCoord, maskedField1)
            cb1 = plt.colorbar(fraction=0.08, pad=0.02, extend=extend)
            plt.xlim(xLims)
            plt.ylim(yLims)
            axis.set_xticklabels([""])
        else:
            plotBasemapPanel(
                maskedField1,
                sector,
                xCoord,
                yCoord,
                lonRange,
                latRange,
                cmap,
                norm,
                interactive,
                extend,
            )
        if centerlabels and len(clim) > 2:
            cb1.set_ticks(0.5 * (clim[:-1] + clim[1:]))
        axis.set_facecolor(landcolor)
        annotateStats(axis, s1Min, s1Max, s1Mean, s1Std, s1RMS, webversion=webversion)
        if len(ylabel + yunits) > 0:
            plt.ylabel(formatting.label(ylabel, yunits))
        if len(title1) > 0:
            if webversion == True:
                axis.annotate(
                    preTitleA + title1,
                    xy=(0.5, 1.14),
                    xycoords="axes fraction",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=12,
                )
            else:
                plt.title(preTitleA + title1)

        axis = plt.subplot(npanels, 1, 2)
        if sector == None or sector == "global":
            plt.pcolormesh(xCoord, yCoord, maskedField2, cmap=cmap, norm=norm)
            if interactive:
                addStatusBar(xCoord, yCoord, maskedField2)
            cb2 = plt.colorbar(fraction=0.08, pad=0.02, extend=extend)
            plt.xlim(xLims)
            plt.ylim(yLims)
            if npanels > 2:
                axis.set_xticklabels([""])
        else:
            plotBasemapPanel(
                maskedField2,
                sector,
                xCoord,
                yCoord,
                lonRange,
                latRange,
                cmap,
                norm,
                interactive,
                extend,
            )
        if centerlabels and len(clim) > 2:
            cb2.set_ticks(0.5 * (clim[:-1] + clim[1:]))
        axis.set_facecolor(landcolor)
        annotateStats(axis, s2Min, s2Max, s2Mean, s2Std, s2RMS, webversion=webversion)
        if len(ylabel + yunits) > 0:
            plt.ylabel(formatting.label(ylabel, yunits))
        if len(title2) > 0:
            if webversion == True:
                axis.annotate(
                    preTitleB + title2,
                    xy=(0.5, titleOffset),
                    xycoords="axes fraction",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=12,
                )
            else:
                plt.title(preTitleB + title2)

    if npanels in [1, 3]:
        axis = plt.subplot(npanels, 1, npanels)
        if sector == None or sector == "global":
            if dcolormap is None:
                dcolormap = chooseColorMap(dMin, dMax)
            if dlim is None and dStd > 0:
                cmap, norm, dextend = cm.chooseColorLevels(
                    dMean - sigma * dStd,
                    dMean + sigma * dStd,
                    dcolormap,
                    clim=dlim,
                    nbins=nbins,
                    extend="both",
                    autocenter=True,
                )
            else:
                cmap, norm, dextend = cm.chooseColorLevels(
                    dMin,
                    dMax,
                    dcolormap,
                    clim=dlim,
                    nbins=nbins,
                    extend=dextend,
                    autocenter=True,
                )
            plt.pcolormesh(
                xCoord, yCoord, maskedField1 - maskedField2, cmap=cmap, norm=norm
            )
            if interactive:
                addStatusBar(xCoord, yCoord, maskedField1 - maskedField2)
            if dextend is None:
                dextend = extend
            cb3 = plt.colorbar(fraction=0.08, pad=0.02, extend=dextend)  # was extend!
            if centerdlabels and len(dlim) > 2:
                cb3.set_ticks(0.5 * (dlim[:-1] + dlim[1:]))
            axis.set_facecolor(landcolor)
            plt.xlim(xLims)
            plt.ylim(yLims)
            annotateStats(axis, dMin, dMax, dMean, dStd, dRMS, webversion=webversion)
            if len(ylabel + yunits) > 0:
                plt.ylabel(formatting.label(ylabel, yunits))
            if len(title3) > 0:
                plt.title(title3)
        elif sector != None:
            # Copy data array, mask, and compute stats / color levels
            maskedDiffField = numpy.ma.array(maskedField1 - maskedField2)
            dMin, dMax, dMean, dStd, dRMS = stats.calc(
                maskedDiffField, areaCopy, debug=debug
            )
            if dcolormap is None:
                dcolormap = chooseColorMap(dMin, dMax, difference=True)
            if dlim is None and dStd > 0:
                cmap, norm, dextend = cm.chooseColorLevels(
                    dMean - sigma * dStd,
                    dMean + sigma * dStd,
                    dcolormap,
                    clim=dlim,
                    nbins=nbins,
                    extend="both",
                    autocenter=True,
                )
            else:
                cmap, norm, dextend = cm.chooseColorLevels(
                    dMin,
                    dMax,
                    dcolormap,
                    clim=dlim,
                    nbins=nbins,
                    extend=dextend,
                    autocenter=True,
                )
            # Set up Basemap projection
            plotBasemapPanel(
                maskedField1 - maskedField2,
                sector,
                xCoord,
                yCoord,
                lonRange,
                latRange,
                cmap,
                norm,
                interactive,
                dextend,
            )
            annotateStats(axis, dMin, dMax, dMean, dStd, dRMS, webversion=webversion)
            if len(ylabel + yunits) > 0:
                plt.ylabel(label(ylabel, yunits))
            if len(title3) > 0:
                if webversion == True:
                    axis.annotate(
                        title3,
                        xy=(0.5, titleOffset),
                        xycoords="axes fraction",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=12,
                    )
                else:
                    plt.title(title3)
        else:
            raise ValueError("Invalid sector specified")

    if webversion:
        plt.subplots_adjust(hspace=hspace)
    if webversion == True:
        fig = plt.gcf()
        fig.text(
            0.5,
            0.02,
            "Generated by m6plot on dora.gfdl.noaa.gov",
            fontsize=9,
            horizontalalignment="center",
        )

    plt.suptitle(suptitle, y=1.0)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    if interactive:
        addInteractiveCallbacks()
    if show:
        plt.show(block=False)

    return plt.gcf()
