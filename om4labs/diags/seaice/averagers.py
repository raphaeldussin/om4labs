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


def leap_year(year, calendar="standard"):
    """Determine if year is a leap year"""
    leap = False
    if (calendar in ["standard", "gregorian", "proleptic_gregorian", "julian"]) and (
        year % 4 == 0
    ):
        leap = True
        if (
            (calendar == "proleptic_gregorian")
            and (year % 100 == 0)
            and (year % 400 != 0)
        ):
            leap = False
        elif (
            (calendar in ["standard", "gregorian"])
            and (year % 100 == 0)
            and (year % 400 != 0)
            and (year < 1583)
        ):
            leap = False
    return leap


def get_dpm(time, calendar="standard"):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar) and month == 2:
            month_length[i] += 1
    return month_length


def annual_cycle(D, var):
    """Compute annual cycle climatology"""
    # Make a DataArray with the number of days in each month, size = len(time)
    if hasattr(D.time, "calendar"):
        calendar = D.time.calendar
    elif hasattr(D.time, "calendar_type"):
        calendar = D.time.calendar_type.lower()
    else:
        calendar = "standard"
    month_length = xr.DataArray(
        get_dpm(D.time.to_index(), calendar=calendar),
        coords=[D.time],
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
    D_weighted = (D[var] * weights).groupby("time.month").sum(dim="time")

    return D_weighted
