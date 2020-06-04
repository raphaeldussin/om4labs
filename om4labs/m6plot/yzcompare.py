def yzcompare(field1, field2, y=None, z=None,
  ylabel=None, yunits=None, zlabel=None, zunits=None,
  splitscale=None,
  title1='', title2='', title3='A - B', addplabel=True, suptitle='',
  clim=None, colormap=None, extend=None, centerlabels=False,
  dlim=None, dcolormap=None, dextend=None, centerdlabels=False,
  nbins=None, landcolor=[.5,.5,.5], sigma=2., webversion=False,
  aspect=None, resolution=None, axis=None, npanels=3,
  ignore=None, save=None, debug=False, show=False, interactive=False):
  """
  Renders n-panel plot of two scalar fields, field1(x,y) and field2(x,y).

  Arguments:
  field1        Scalar 2D array to be plotted and compared to field2.
  field2        Scalar 2D array to be plotted and compared to field1.
  y             y coordinate (1D array). If y is the same size as field then y is treated as
                the cell center coordinates.
  z             z coordinate (1D or 2D array). If z is the same size as field then z is treated as
                the cell center coordinates.
  ylabel        The label for the y axis. Default 'Latitude'.
  yunits        The units for the y axis. Default 'degrees N'.
  zlabel        The label for the z axis. Default 'Elevation'.
  zunits        The units for the z axis. Default 'm'.
  splitscale    A list of depths to define equal regions of projection in the vertical, e.g. [0.,-1000,-6500]
  title1        The title to place at the top of panel 1. Default ''.
  title2        The title to place at the top of panel 1. Default ''.
  title3        The title to place at the top of panel 1. Default 'A-B'.
  addplabel     Adds a 'A:' or 'B:' to the title1 and title2. Default True.
  suptitle      The super-title to place at the top of the figure. Default ''.
  clim          A tuple of (min,max) color range OR a list of contour levels for the field plots. Default None.
  sigma         Sigma range for difference plot autocolor levels. Default is to span a 2. sigma range
  colormap      The name of the colormap to use for the field plots. Default None.
  extend        Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels  If True, will move the colorbar labels to the middle of the interval. Default False.
  dlim          A tuple of (min,max) color range OR a list of contour levels for the difference plot. Default None.
  dcolormap     The name of the colormap to use for the differece plot. Default None.
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

  if (field1.shape)!=(field2.shape): raise Exception('field1 and field2 must be the same shape')

  # Create coordinates if not provided
  ylabel, yunits, zlabel, zunits = createYZlabels(y, z, ylabel, yunits, zlabel, zunits)
  if debug: print('y,z label/units=',ylabel,yunits,zlabel,zunits)
  if len(y)==z.shape[-1]: y= expand(y)
  elif len(y)==z.shape[-1]+1: y= y
  else: raise Exception('Length of y coordinate should be equal or 1 longer than horizontal length of z')
  if ignore is not None: maskedField1 = numpy.ma.masked_array(field1, mask=[field1==ignore])
  else: maskedField1 = field1.copy()
  yCoord, zCoord, field1 = m6toolbox.section2quadmesh(y, z, maskedField1)

  # Diagnose statistics
  yzWeighting = yzWeight(y, z)
  s1Min, s1Max, s1Mean, s1Std, s1RMS = myStats(maskedField1, yzWeighting, debug=debug)
  if ignore is not None: maskedField2 = numpy.ma.masked_array(field2, mask=[field2==ignore])
  else: maskedField2 = field2.copy()
  yCoord, zCoord, field2 = m6toolbox.section2quadmesh(y, z, maskedField2)
  s2Min, s2Max, s2Mean, s2Std, s2RMS = myStats(maskedField2, yzWeighting, debug=debug)
  dMin, dMax, dMean, dStd, dRMS = myStats(maskedField1 - maskedField2, yzWeighting, debug=debug)
  dRxy = corr(maskedField1 - s1Mean, maskedField2 - s2Mean, yzWeighting)
  s12Min = min(s1Min, s2Min); s12Max = max(s1Max, s2Max)
  xLims = numpy.amin(yCoord), numpy.amax(yCoord); yLims = boundaryStats(zCoord)
  if debug:
    print('s1: min, max, mean =', s1Min, s1Max, s1Mean)
    print('s2: min, max, mean =', s2Min, s2Max, s2Mean)
    print('s12: min, max =', s12Min, s12Max)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): cBins=35
  else: cBins=nbins
  if nbins is None and (dlim is None or len(dlim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(s12Min, s12Max)
  cmap, norm, extend = chooseColorLevels(s12Min, s12Max, colormap, clim=clim, nbins=cBins, extend=extend)

  def annotateStats(axis, sMin, sMax, sMean, sStd, sRMS, webversion=False):
    if webversion == True: fontsize=9
    else: fontsize=10
    axis.annotate('max=%.5g\nmin=%.5g'%(sMax,sMin), xy=(0.0,1.025), xycoords='axes fraction', \
                  verticalalignment='bottom', fontsize=fontsize)
    if sMean is not None:
      axis.annotate('mean=%.5g\nrms=%.5g'%(sMean,sRMS), xy=(1.0,1.025), xycoords='axes fraction', \
                    verticalalignment='bottom', horizontalalignment='right', fontsize=fontsize)
      axis.annotate(' sd=%.5g\n'%(sStd), xy=(1.0,1.025), xycoords='axes fraction', verticalalignment='bottom', \
                    horizontalalignment='left', fontsize=fontsize)

  if addplabel: preTitleA = 'A: '; preTitleB = 'B: '
  else: preTitleA = ''; preTitleB = ''

  if axis is None:
    setFigureSize(aspect, resolution, npanels=npanels, debug=debug)
    #plt.gcf().subplots_adjust(left=.13, right=.94, wspace=0, bottom=.05, top=.94, hspace=0.15)

  if npanels in [2, 3]:
    axis = plt.subplot(npanels,1,1)
    plt.pcolormesh(yCoord, zCoord, field1, cmap=cmap, norm=norm)
    if interactive: addStatusBar(yCoord, zCoord, field1)
    cb1 = plt.colorbar(fraction=.08, pad=0.02, extend=extend)
    if centerlabels and len(clim)>2: cb1.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
    axis.set_facecolor(landcolor)
    if splitscale is not None:
      for zzz in splitscale[1:-1]: plt.axhline(zzz,color='k',linestyle='--')
      axis.set_yscale('splitscale', zval=splitscale)
    plt.xlim( xLims ); plt.ylim( yLims )
    annotateStats(axis, s1Min, s1Max, s1Mean, s1Std, s1RMS, webversion=webversion)
    axis.set_xticklabels([''])
    if len(zlabel+zunits)>0: plt.ylabel(label(zlabel, zunits))
    if len(title1)>0: plt.title(preTitleA+title1)

    axis = plt.subplot(npanels,1,2)
    plt.pcolormesh(yCoord, zCoord, field2, cmap=cmap, norm=norm)
    if interactive: addStatusBar(yCoord, zCoord, field2)
    cb2 = plt.colorbar(fraction=.08, pad=0.02, extend=extend)
    if centerlabels and len(clim)>2: cb2.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
    axis.set_facecolor(landcolor)
    if splitscale is not None:
      for zzz in splitscale[1:-1]: plt.axhline(zzz,color='k',linestyle='--')
      axis.set_yscale('splitscale', zval=splitscale)
    plt.xlim( xLims ); plt.ylim( yLims )
    annotateStats(axis, s2Min, s2Max, s2Mean, s2Std, s2RMS, webversion=webversion)
    if npanels>2: axis.set_xticklabels([''])
    if len(zlabel+zunits)>0: plt.ylabel(label(zlabel, zunits))
    if len(title2)>0: plt.title(preTitleB+title2)

  if npanels in [1, 3]:
    axis = plt.subplot(npanels,1,npanels)
    if dcolormap is None: dcolormap = chooseColorMap(dMin, dMax)
    if dlim is None and dStd>0:
      cmap, norm, dextend = chooseColorLevels(dMean-sigma*dStd, dMean+sigma*dStd, dcolormap, clim=dlim, nbins=nbins, extend='both', autocenter=True)
    else:
      cmap, norm, dextend = chooseColorLevels(dMin, dMax, dcolormap, clim=dlim, nbins=nbins, extend=dextend, autocenter=True)
    plt.pcolormesh(yCoord, zCoord, field1 - field2, cmap=cmap, norm=norm)
    if interactive: addStatusBar(yCoord, zCoord, field1 - field2)
    cb3 = plt.colorbar(fraction=.08, pad=0.02, extend=dextend)
    if centerdlabels and len(dlim)>2: cb3.set_ticks(  0.5*(dlim[:-1]+dlim[1:]) )
    axis.set_facecolor(landcolor)
    if splitscale is not None:
      for zzz in splitscale[1:-1]: plt.axhline(zzz,color='k',linestyle='--')
      axis.set_yscale('splitscale', zval=splitscale)
    plt.xlim( xLims ); plt.ylim( yLims )
    annotateStats(axis, dMin, dMax, dMean, dStd, dRMS)
    if len(zlabel+zunits)>0: plt.ylabel(label(zlabel, zunits))

  axis.annotate(' r(A,B)=%.5g\n'%(dRxy), xy=(1.0,-1.07), xycoords='axes fraction', verticalalignment='top', horizontalalignment='center', fontsize=10)
  if len(ylabel+yunits)>0: plt.xlabel(label(ylabel, yunits))
  if len(title3)>0: plt.title(title3)
  if len(suptitle)>0: plt.suptitle(suptitle)

  if save is not None: plt.savefig(save)
  if interactive: addInteractiveCallbacks()
  if show: plt.show(block=False)
