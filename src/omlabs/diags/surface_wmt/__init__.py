from .surface_wmt import parse, read, calculate, plot, run, parse_and_run

__description__ = "Plots global watermass transformation (across density surfaces) from fluxes of heat, salt, and freshwater at the ocean surface"
__ppstreams__ = [
    "ocean_monthly/av",
]
__ppvars__ = ["hfds","wfo","sfdsi","tos","sos"]
