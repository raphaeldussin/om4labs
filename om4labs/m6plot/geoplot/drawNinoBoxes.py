def drawNinoBoxes(m,region='all'):
  '''
  Function to draw ENSO region boxes on a basemap instance
  '''
  if region == 'nino4' or region == 'all':
    polyLon = [-200., -200., -150., -150., -200.]
    polyLat = [-5., 5., 5., -5., -5.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='k',linewidth=2.0)
  if region == 'nino3' or region == 'all':
    polyLon = [-150., -150., -90., -90., -150.]
    polyLat = [-5., 5., 5., -5., -5.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='k',linewidth=2.0)
  if region == 'nino34' or region == 'all':
    polyLon = [-170., -170., -120., -120., -170.]
    polyLat = [-5., 5., 5., -5., -5.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='r',linestyle='dashed',linewidth=2.0)
  if region == 'nino12' or region == 'all':
    polyLon = [-90., -90., -80., -80., -90.]
    polyLat = [-10., 0., 0., -10., -10.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='k',linewidth=2.0)
