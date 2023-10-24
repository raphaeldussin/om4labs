from .enso import parse, read, calculate, plot, run, parse_and_run

__description__ = "Plots power spectra and wavelets for NINO regions"
__ppstreams__ = [
    "ocean_monthly_1x1deg/ts",
]
__ppvars__ = ["tos"]
