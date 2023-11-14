from .mld import parse, read, calculate, plot, run, parse_and_run

__description__ = "Mixed Layer Depth bias maps"
__ppstreams__ = [
    "ocean_monthly/av",
]
__ppvars__ = ["MLD_003","MLD_EN1","MLD_EN2","MLD_EN3"]
