import numpy


def expand(a):
    """
  Expands a vector by one element, averaging the data to the middle columns and
  extrapolating for the first and last rows. Needed for shifting coordinates
  from centers to corners.
  """
    b = numpy.zeros((len(a) + 1))
    b[1:-1] = 0.5 * (a[:-1] + a[1:])
    b[0] = a[0] + 0.5 * (a[0] - a[1])
    b[-1] = a[-1] + 0.5 * (a[-1] - a[-2])
    return b


def expandI(a):
    """
  Expands an array by one column, averaging the data to the middle columns and
  extrapolating for the first and last columns. Needed for shifting coordinates
  from centers to corners.
  """
    nj, ni = a.shape
    b = numpy.zeros((nj, ni + 1))
    b[:, 1:-1] = 0.5 * (a[:, :-1] + a[:, 1:])
    b[:, 0] = a[:, 0] + 0.5 * (a[:, 0] - a[:, 1])
    b[:, -1] = a[:, -1] + 0.5 * (a[:, -1] - a[:, -2])
    return b


def expandJ(a):
    """
  Expands an array by one row, averaging the data to the middle columns and
  extrapolating for the first and last rows. Needed for shifting coordinates
  from centers to corners.
  """
    nj, ni = a.shape
    b = numpy.zeros((nj + 1, ni))
    b[1:-1, :] = 0.5 * (a[:-1, :] + a[1:, :])
    b[0, :] = a[0, :] + 0.5 * (a[0, :] - a[1, :])
    b[-1, :] = a[-1, :] + 0.5 * (a[-1, :] - a[-2, :])
    return b


def boundaryStats(a):
    """
  Returns the minimum and maximum values of a only on the boundaries of the array.
  """
    amin = numpy.amin(a[0, :])
    amin = min(amin, numpy.amin(a[1:, -1]))
    amin = min(amin, numpy.amin(a[-1, :-1]))
    amin = min(amin, numpy.amin(a[1:-1, 0]))
    amax = numpy.amax(a[0, :])
    amax = max(amax, numpy.amax(a[1:, -1]))
    amax = max(amax, numpy.amax(a[-1, :-1]))
    amax = max(amax, numpy.amax(a[1:-1, 0]))
    return amin, amax
