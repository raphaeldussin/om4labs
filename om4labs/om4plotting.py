import io

try:
    from om4labs import m6plot
except ImportError:
    import m6plot


def plot_xydiff(x, y, slice1, slice2, diff_kwargs, stream=False):
    """ make difference plot """
    if stream:
        img = io.BytesIO()
        diff_kwargs["save"] = img

    fig = m6plot.xyplot(slice1 - slice2, x, y, **diff_kwargs)

    # if stream:
    #    return img
    # else:
    print(fig)
    return fig


def plot_xycompare(x, y, slice1, slice2, compare_kwargs, stream=False):
    """ make 3 panel compare plot """
    if stream:
        img = io.BytesIO()
        compare_kwargs["img"] = img

    fig = m6plot.xycompare(slice1, slice2, x, y, **compare_kwargs)

    # if stream:
    #    return img
    # else:
    return fig
