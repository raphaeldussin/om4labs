def drawNinoBoxes(m, region="all"):
    """
  Function to draw ENSO region boxes on a basemap instance
  """
    if region == "nino4" or region == "all":
        polyLon = [-200.0, -200.0, -150.0, -150.0, -200.0]
        polyLat = [-5.0, 5.0, 5.0, -5.0, -5.0]
        polyX, polyY = m(polyLon, polyLat)
        m.plot(polyX, polyY, marker=None, color="k", linewidth=2.0)
    if region == "nino3" or region == "all":
        polyLon = [-150.0, -150.0, -90.0, -90.0, -150.0]
        polyLat = [-5.0, 5.0, 5.0, -5.0, -5.0]
        polyX, polyY = m(polyLon, polyLat)
        m.plot(polyX, polyY, marker=None, color="k", linewidth=2.0)
    if region == "nino34" or region == "all":
        polyLon = [-170.0, -170.0, -120.0, -120.0, -170.0]
        polyLat = [-5.0, 5.0, 5.0, -5.0, -5.0]
        polyX, polyY = m(polyLon, polyLat)
        m.plot(polyX, polyY, marker=None, color="r", linestyle="dashed", linewidth=2.0)
    if region == "nino12" or region == "all":
        polyLon = [-90.0, -90.0, -80.0, -80.0, -90.0]
        polyLat = [-10.0, 0.0, 0.0, -10.0, -10.0]
        polyX, polyY = m(polyLon, polyLat)
        m.plot(polyX, polyY, marker=None, color="k", linewidth=2.0)
