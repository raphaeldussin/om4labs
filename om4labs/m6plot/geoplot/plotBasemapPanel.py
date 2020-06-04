def plotBasemapPanel(field, sector, xCoord, yCoord, lonRange, latRange, cmap, norm, interactive, extend):
  if sector == 'arctic':  m = Basemap(projection='npstere',boundinglat=60,lon_0=-120,resolution='l')
  elif sector == 'shACC': m = Basemap(projection='spstere',boundinglat=-45,lon_0=-120,resolution='l')
  else:  m = Basemap(projection='mill',lon_0=-120.,resolution='l',llcrnrlon=lonRange[0], \
             llcrnrlat=latRange[0], urcrnrlon=lonRange[1],urcrnrlat=latRange[1])
  m.drawmapboundary(fill_color='0.85')
  im0 = m.pcolormesh(numpy.minimum(xCoord,60.),yCoord,(field),shading='flat',cmap=cmap,norm=norm,latlon=True)
  m.drawcoastlines()
  if interactive: addStatusBar(xCoord, yCoord, field)
  if extend is None: extend = extend
  cb1 = m.colorbar(pad=0.15, extend=extend)
  if sector == 'tropPac': drawNinoBoxes(m)
