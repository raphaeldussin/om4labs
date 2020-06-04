import numpy
import numpy.matlib

def yzWeight(y, z):
  """
  Calculates the weights to use when calculating the statistics of a y-z section.

  y(nj+1) is a 1D vector of column edge positions and z(nk+1,nj) is the interface
  elevations of each column. Returns weight(nk,nj).
  """
  dz = z[:-1,:] - z[1:,:]
  return numpy.matlib.repmat(y[1:] - y[:-1], dz.shape[0], 1) * dz
