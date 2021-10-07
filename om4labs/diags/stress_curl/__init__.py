from .stress_curl import parse, read, calculate, plot, run, parse_and_run

__description__ = "Maps the curl of the stress acting on the surface ocean"
__ppstreams__ = [
    "ocean_annual/ts",
    "ocean_monthly/ts",
]
__ppvars__ = ["tauuo", "tauvo"]
