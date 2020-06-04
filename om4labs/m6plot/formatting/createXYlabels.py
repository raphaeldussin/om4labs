def createXYlabels(x, y, xlabel, xunits, ylabel, yunits):
  """
  Checks that x and y labels are appropriate and tries to make some if they are not.
  """
  if x is None:
    if xlabel is None: xlabel='i'
    if xunits is None: xunits=''
  else:
    if xlabel is None: xlabel='Longitude'
    #if xunits is None: xunits=u'\u00B0E'
    if xunits is None: xunits=r'$\degree$E'
  if y is None:
    if ylabel is None: ylabel='j'
    if yunits is None: yunits=''
  else:
    if ylabel is None: ylabel='Latitude'
    #if yunits is None: yunits=u'\u00B0N'
    if yunits is None: yunits=r'$\degree$N'
  return xlabel, xunits, ylabel, yunits
