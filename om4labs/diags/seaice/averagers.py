import calendar
import xarray as xr
import numpy as np


def annual_cycle(ds, var):
    """Compute annual cycle climatology"""
    # Make a DataArray with the number of days in each month, size = len(time)
    if hasattr(ds.time, "calendar"):
        cal = ds.time.calendar
    elif hasattr(ds.time, "calendar_type"):
        cal = ds.time.calendar_type.lower()
    else:
        cal = "standard"

    if cal.lower() in ["noleap", "365day"]:
        # always calculate days in month based on year 1 (non-leap year)
        month_length = [calendar.monthrange(1, x.month)[1] for x in ds.time.to_index()]
    else:
        # use real year/month combo to calculate days in month
        month_length = [
            calendar.monthrange(x.year, x.month)[1] for x in ds.time.to_index()
        ]

    month_length = xr.DataArray(month_length, coords=[ds.time], name="month_length")

    # Calculate the weights by grouping by 'time.season'.
    # Conversion to float type ('astype(float)') only necessary for Python 2.x
    weights = (
        month_length.groupby("time.month") / month_length.groupby("time.month").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.month").sum().values, np.ones(12))

    # Calculate the weighted average
    ds_weighted = (ds[var] * weights).groupby("time.month").sum(dim="time")

    return ds_weighted
