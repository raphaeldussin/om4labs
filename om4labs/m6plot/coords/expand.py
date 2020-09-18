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
