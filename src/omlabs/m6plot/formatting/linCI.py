import numpy as np


def linCI(min, max, ci, *args):
    """
    Returns list of linearly spaced contour intervals from min to max with spacing ci.
    Unline np.arange this max is included IF max = min + ci*N for an integer N.
    """
    if len(args):
        return np.concatenate((np.arange(min, max + ci, ci), linCI(*args)))
    return np.arange(min, max + ci, ci)
