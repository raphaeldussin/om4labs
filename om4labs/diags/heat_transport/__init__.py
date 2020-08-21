from .heat_transport import parse, read, calculate, plot, run, parse_and_run

__description__ = "Plots global and basin-average poleward heat transport"
__ppstreams__ = [
    "ocean_month_z_d2_refined/av",
    "ocean_monthly_z_d2/ts",
    "ocean_annual_z/av",
]
__ppvars__ = ["vmo"]
