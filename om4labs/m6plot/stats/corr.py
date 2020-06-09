import numpy
import math


def corr(s1, s2, area):
    """
  Calculates the correlation coefficient between s1 and s2, assuming s1 and s2 have
  not mean. That is s1 = S - mean(S), etc.
  """
    weight = area.copy()
    if not numpy.ma.getmask(s1).any() == numpy.ma.nomask:
        weight[s1.mask] = 0.0
    sumArea = numpy.ma.sum(weight)
    v1 = numpy.ma.sum(weight * (s1 ** 2)) / sumArea
    v2 = numpy.ma.sum(weight * (s2 ** 2)) / sumArea
    if v1 == 0 or v2 == 0:
        return numpy.NaN
    rxy = numpy.ma.sum(weight * (s1 * s2)) / sumArea / math.sqrt(v1 * v2)
    return rxy
