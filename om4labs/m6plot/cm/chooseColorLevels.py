import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import MaxNLocator

def chooseColorLevels(
    sMin,
    sMax,
    colorMapName,
    clim=None,
    nbins=None,
    steps=[1, 2, 2.5, 5, 10],
    extend=None,
    logscale=False,
    autocenter=False,
):
    """
  If nbins is a positive integer, choose sensible color levels with nbins colors.
  If clim is a 2-element tuple, create color levels within the clim range
  or if clim is a vector, use clim as contour levels.
  If clim provides more than 2 color interfaces, nbins must be absent.
  If clim is absent, the sMin,sMax are used as the color range bounds.
  If autocenter is True and clim is None then the automatic color levels are centered.

  Returns cmap, norm and extend.
  """
    if nbins is None and clim is None:
        raise Exception("At least one of clim or nbins is required.")
    if clim is not None:
        if len(clim) < 2:
            raise Exception("clim must be at least 2 values long.")
        if nbins is None and len(clim) == 2:
            raise Exception(
                "nbins must be provided when clims specifies a color range."
            )
        if nbins is not None and len(clim) > 2:
            raise Exception(
                "nbins cannot be provided when clims specifies color levels."
            )
    if clim is None:
        if autocenter:
            levels = MaxNLocator(nbins=nbins, steps=steps).tick_values(
                min(sMin, -sMax), max(sMax, -sMin)
            )
        else:
            levels = MaxNLocator(nbins=nbins, steps=steps).tick_values(sMin, sMax)
    elif len(clim) == 2:
        levels = MaxNLocator(nbins=nbins, steps=steps).tick_values(clim[0], clim[1])
    else:
        levels = clim

    nColors = len(levels) - 1
    if extend is None:
        if sMin < levels[0] and sMax > levels[-1]:
            extend = "both"  # ; eColors=[1,1]
        elif sMin < levels[0] and sMax <= levels[-1]:
            extend = "min"  # ; eColors=[1,0]
        elif sMin >= levels[0] and sMax > levels[-1]:
            extend = "max"  # ; eColors=[0,1]
        else:
            extend = "neither"  # ; eColors=[0,0]
    eColors = [0, 0]
    if extend in ["both", "min"]:
        eColors[0] = 1
    if extend in ["both", "max"]:
        eColors[1] = 1

    cmap = plt.get_cmap(colorMapName)  # ,lut=nColors+eColors[0]+eColors[1])
    # cmap0 = cmap(0.)
    # cmap1 = cmap(1.)
    # cmap = ListedColormap(cmap(range(eColors[0],nColors+1-eColors[1]+eColors[0])))#, N=nColors)
    # if eColors[0]>0: cmap.set_under(cmap0)
    # if eColors[1]>0: cmap.set_over(cmap1)
    if logscale:
        norm = LogNorm(vmin=levels[0], vmax=levels[-1])
    else:
        norm = BoundaryNorm(levels, ncolors=cmap.N)
    return cmap, norm, extend
