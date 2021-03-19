import warnings
import palettable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from om4labs import m6plot

warnings.filterwarnings("ignore", message=".*csr_matrix.*")
warnings.filterwarnings("ignore", message=".*dates out of range.*")


def plot_rho(dset, otsfn, label=None):
    print(dset)
    print(otsfn)

    # setup figure handle
    fig = plt.figure(figsize=(8.5, 11))

    return fig