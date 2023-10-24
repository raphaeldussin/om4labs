import numpy


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
