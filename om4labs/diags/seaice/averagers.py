import calendar
import xarray as xr
import numpy as np

dpm = {
    "noleap": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "365_day": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "standard": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "proleptic_gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "all_leap": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "366_day": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "360_day": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
}


def leap_year(year, cal="standard"):
    """Determine if year is a leap year"""
    year = int(year)
    if cal in ["standard", "gregorian", "proleptic_gregorian", "julian"]:
        return calendar.isleap(year)
    else:
        return False


def get_dpm(time, cal="standard"):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[cal]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, cal=cal) and month == 2:
            month_length[i] += 1
    return month_length


def annual_cycle(ds, var):
    """Compute annual cycle climatology"""
    # Make a DataArray with the number of days in each month, size = len(time)
    if hasattr(ds.time, "calendar"):
        cal = ds.time.calendar
    elif hasattr(ds.time, "calendar_type"):
        cal = ds.time.calendar_type.lower()
    else:
        cal = "standard"
    month_length = xr.DataArray(
        get_dpm(ds.time.to_index(), cal=cal),
        coords=[ds.time],
        name="month_length",
    )

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
