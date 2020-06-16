# -*- coding: utf-8 -*-
"""m6plot.geoplot - routines to use a higher level mapping package

*** this uses Basemap, which is depreciated; needs to be ported to cartopy ***
"""

__all__ = [
    "sectorRanges",
]

for func in __all__:
    exec(f"from .{func} import *")
