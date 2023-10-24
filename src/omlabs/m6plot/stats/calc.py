import math
import numpy


def calc(s, area, s2=None, debug=False):
    """
    Calculates mean, standard deviation and root-mean-square of s.
    """
    sMin = numpy.ma.min(s)
    sMax = numpy.ma.max(s)
    if area is None:
        return sMin, sMax, None, None, None
    weight = area.copy()
    if debug:
        print("stats: sum(area) =", numpy.ma.sum(weight))
    if not numpy.ma.getmask(s).any() == numpy.ma.nomask:
        weight[s.mask] = 0.0
    sumArea = numpy.ma.sum(weight)
    if debug:
        print("stats: sum(area) =", sumArea, "after masking")
    if debug:
        print("stats: sum(s) =", numpy.ma.sum(s))
    if debug:
        print("stats: sum(area*s) =", numpy.ma.sum(weight * s))
    mean = numpy.ma.sum(weight * s) / sumArea
    std = math.sqrt(numpy.ma.sum(weight * ((s - mean) ** 2)) / sumArea)
    rms = math.sqrt(numpy.ma.sum(weight * (s ** 2)) / sumArea)
    if debug:
        print("stats: mean(s) =", mean)
    if debug:
        print("stats: std(s) =", std)
    if debug:
        print("stats: rms(s) =", rms)
    return sMin, sMax, mean, std, rms
