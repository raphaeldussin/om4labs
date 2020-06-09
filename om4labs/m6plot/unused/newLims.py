def newLims(cur_xlim, cur_ylim, cursor, xlim, ylim, scale_factor):
    cur_xrange = (cur_xlim[1] - cur_xlim[0]) * 0.5
    cur_yrange = (cur_ylim[1] - cur_ylim[0]) * 0.5
    xdata = cursor[0]
    ydata = cursor[1]
    new_xrange = cur_xrange * scale_factor
    new_yrange = cur_yrange * scale_factor
    xdata = min(max(xdata, xlim[0] + new_xrange), xlim[1] - new_xrange)
    xL = max(xlim[0], xdata - new_xrange)
    xR = min(xlim[1], xdata + new_xrange)
    if ylim[1] > ylim[0]:
        ydata = min(max(ydata, ylim[0] + new_yrange), ylim[1] - new_yrange)
        yL = max(ylim[0], ydata - new_yrange)
        yR = min(ylim[1], ydata + new_yrange)
    else:
        ydata = min(max(ydata, ylim[1] - new_yrange), ylim[0] + new_yrange)
        yR = max(ylim[1], ydata + new_yrange)
        yL = min(ylim[0], ydata - new_yrange)
    if (
        xL == cur_xlim[0]
        and xR == cur_xlim[1]
        and yL == cur_ylim[0]
        and yR == cur_ylim[1]
    ):
        return (None, None), (None, None)
    return (xL, xR), (yL, yR)
