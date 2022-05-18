from .generic_depth_time_drift import parse, read, calculate, plot, run, parse_and_run

__description__ = "Generic layer-average drift diagnostic"
__ppstreams__ = [
    "ocean_annual_z/ts",
]
__ppvars__ = ["thetao_xyave", "so_xyave"]
