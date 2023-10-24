import io

try:
    from om4labs import m6plot
except ImportError:
    import m6plot


def plot_xydiff(x, y, slice1, slice2, diff_kwargs, interactive=False, stream=False):
    """make difference plot"""
    if stream:
        img = io.BytesIO()
        diff_kwargs["save"] = img

    fig = m6plot.xyplot(slice1 - slice2, x, y, interactive=interactive, **diff_kwargs)

    return fig


def plot_xycompare(
    x, y, slice1, slice2, compare_kwargs, interactive=False, stream=False
):
    """make 3 panel compare plot"""
    if stream:
        img = io.BytesIO()
        compare_kwargs["save"] = img

    fig = m6plot.xycompare(
        slice1, slice2, x, y, interactive=interactive, **compare_kwargs
    )

    return fig


def plot_yzdiff(y, z, slice1, slice2, diff_kwargs, interactive=False, stream=False):
    """make difference plot"""
    if stream:
        img = io.BytesIO()
        diff_kwargs["save"] = img

    fig = m6plot.yzplot(slice1 - slice2, y, z, interactive=interactive, **diff_kwargs)

    return fig


def plot_yzcompare(
    y, z, slice1, slice2, compare_kwargs, interactive=False, stream=False
):
    """make 3 panel compare plot"""
    if stream:
        img = io.BytesIO()
        compare_kwargs["save"] = img

    fig = m6plot.yzcompare(
        slice1, slice2, y, z, interactive=interactive, **compare_kwargs
    )

    return fig
