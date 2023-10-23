from .heat_transport import parse, read, calculate, plot, run, parse_and_run

__description__ = "Plots global and basin-average poleward heat transport"
__ppstreams__ = [
    "ocean_monthly/av",
]
__ppvars__ = ["T_ady_2d"]
