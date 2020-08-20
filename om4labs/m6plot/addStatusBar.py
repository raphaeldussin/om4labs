import matplotlib.pyplot as plt
import numpy as np


def addStatusBar(xCoord, yCoord, zData):
    """
  Reformats status bar message
  """

    class hiddenStore:
        def __init__(self, axis):
            self.axis = axis
            self.xMin, self.xMax = axis.get_xlim()
            self.yMin, self.yMax = axis.get_ylim()

    save = hiddenStore(plt.gca())

    def statusMessage(x, y):
        # THIS NEEDS TESTING FOR ACCURACY, ESPECIALLY IN YZ PLOTS -AJA
        if len(xCoord.shape) == 1 and len(yCoord.shape) == 1:
            # -2 needed because of coords are for vertices and need to be averaged to centers
            i = min(
                range(len(xCoord) - 2),
                key=lambda l: abs((xCoord[l] + xCoord[l + 1]) / 2.0 - x),
            )
            j = min(
                range(len(yCoord) - 2),
                key=lambda l: abs((yCoord[l] + yCoord[l + 1]) / 2.0 - y),
            )
        elif len(xCoord.shape) == 1 and len(yCoord.shape) == 2:
            i = min(
                range(len(xCoord) - 2),
                key=lambda l: abs((xCoord[l] + xCoord[l + 1]) / 2.0 - x),
            )
            j = min(
                range(len(yCoord[:, i]) - 1),
                key=lambda l: abs((yCoord[l, i] + yCoord[l + 1, i]) / 2.0 - y),
            )
        elif len(xCoord.shape) == 2 and len(yCoord.shape) == 2:
            idx = np.abs(
                np.fabs(
                    xCoord[0:-1, 0:-1]
                    + xCoord[1:, 1:]
                    + xCoord[0:-1, 1:]
                    + xCoord[1:, 0:-1]
                    - 4 * x
                )
                + np.fabs(
                    yCoord[0:-1, 0:-1]
                    + yCoord[1:, 1:]
                    + yCoord[0:-1, 1:]
                    + yCoord[1:, 0:-1]
                    - 4 * y
                )
            ).argmin()
            j, i = np.unravel_index(idx, zData.shape)
        else:
            raise Exception("Combindation of coordinates shapes is VERY UNUSUAL!")
        if not i is None:
            val = zData[j, i]
            if val is np.ma.masked:
                return "x,y=%.3f,%.3f  f(%i,%i)=NaN" % (x, y, i + 1, j + 1)
            else:
                return "x,y=%.3f,%.3f  f(%i,%i)=%g" % (x, y, i + 1, j + 1, val)
        else:
            return "x,y=%.3f,%.3f" % (x, y)

    plt.gca().format_coord = statusMessage
