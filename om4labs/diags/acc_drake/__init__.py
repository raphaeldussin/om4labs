from .acc_timeseries import parse, read, calculate, plot, run, parse_and_run

__description__ = (
    "Plots time series of total volume transport through the Drake Passage"
)
__ppstreams__ = [
    "ocean_month_z_d2_refined/av",
    "ocean_monthly_z_d2/ts",
    "ocean_annual_z/av",
]
__ppvars__ = ["umo"]
