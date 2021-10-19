from .thermocline import parse, read, calculate, plot, run, parse_and_run

__description__ = "Evaluate thermocline strength and depth relative to WOA"
__ppstreams__ = [
    "ocean_annual_z_1x1deg/av",
]
__ppvars__ = ["thetao"]
