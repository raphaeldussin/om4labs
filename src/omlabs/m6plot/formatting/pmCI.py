import numpy as np
from .linCI import *


def pmCI(min, max, ci, *args):
    """
    Returns list of linearly spaced contour intervals from -max to -min then min to max with spacing ci.
    Unline np.arange this max is included IF max = min + ci*N for an integer N.
    """
    ci = linCI(min, max, ci, *args)
    if ci[0] > 0:
        return np.concatenate((-ci[::-1], ci))
    else:
        return np.concatenate((-ci[::-1], ci[1:]))
