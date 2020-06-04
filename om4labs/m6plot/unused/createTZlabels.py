def createTZlabels(t, z, tlabel, tunits, zlabel, zunits):
  """
  Checks that y and z labels are appropriate and tries to make some if they are not.
  """
  if t is None:
    if tlabel is None: tlabel='t'
    if tunits is None: tunits=''
  else:
    if tlabel is None: tlabel='Time'
    if tunits is None: tunits=''
  if z is None:
    if zlabel is None: zlabel='k'
    if zunits is None: zunits=''
  else:
    if zlabel is None: zlabel='Elevation'
    if zunits is None: zunits='m'
  return tlabel, tunits, zlabel, zunits
