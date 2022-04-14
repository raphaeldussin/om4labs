from .depth_time_temperature_drift import (
    parse,
    read,
    calculate,
    plot,
    run,
    parse_and_run,
)

__description__ = "Layer mean potential temperature drift vs. time"
__ppstreams__ = [
    "ocean_annual_z/ts",
]
__ppvars__ = ["thetao_xyave"]
