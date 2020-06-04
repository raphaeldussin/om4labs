import numpy
import numpy.matlib
from ..coords import *

def createXYcoords(s, x, y):
  """
  Checks that x and y are appropriate 2D corner coordinates
  and tries to make some if they are not.
  """
  nj, ni = s.shape
  if x is None: xCoord = numpy.arange(0., ni+1)
  else: xCoord = numpy.ma.filled(x, 0.)
  if y is None: yCoord = numpy.arange(0., nj+1)
  else: yCoord = numpy.ma.filled(y, 0.)

  # Turn coordinates into 2D arrays if 1D arrays were provided
  if len(xCoord.shape)==1:
    nxy = yCoord.shape
    xCoord = numpy.matlib.repmat(xCoord, nxy[0], 1)
  nxy = xCoord.shape
  if len(yCoord.shape)==1: yCoord = numpy.matlib.repmat(yCoord.T, nxy[-1], 1).T
  if xCoord.shape!=yCoord.shape: raise Exception('The shape of coordinates are mismatched!')

  # Create corner coordinates from center coordinates is center coordinates were provided
  if xCoord.shape!=yCoord.shape: raise Exception('The shape of coordinates are mismatched!')
  if s.shape==xCoord.shape:
    xCoord = expandJ( expandI( xCoord ) )
    yCoord = expandJ( expandI( yCoord ) )
  return xCoord, yCoord

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
