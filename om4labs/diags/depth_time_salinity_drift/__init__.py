from .depth_time_salinity_drift import parse, read, calculate, plot, run, parse_and_run

__description__ = "Layer mean practical salinity drift vs. time"
__ppstreams__ = [
    "ocean_annual_z/ts",
]
__ppvars__ = ["so_xyave"]
