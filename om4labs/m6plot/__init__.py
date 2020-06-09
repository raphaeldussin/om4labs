# -*- coding: utf-8 -*-
"""m6plot - a collection of routines for plotting MOM6 output

This module contains a collection of utilities to create lightweight
plots of MOM6 output. Plots include 2-dimensional maps as well as 
3-dimenstional sections.
"""

__version__ = "0.0.0"
funcs = [
    "addInteractiveCallbacks",
    "addStatusBar",
    "cm",
    "xyplot",
    "xycompare",
    "yzplot",
    #'yzcompare',
    #'ztplot',
    #'brownblue_cmap',
    #'cmoceanRegisterColormaps',
    #'createTZlabels',
    #'createYZlabels',
    #'drawNinoBoxes',
    #'linCI',
    #'newLims',
    #'parula_cmap',
    #'plotBasemapPanel',
    #'pmCI',
    #'regionalMasking',
]

for func in funcs:
    exec(f"from .{func} import *")

mods = [
    "cm",
    "coords",
    "formatting",
    "geoplot",
    "stats",
]

for mod in mods:
    exec(f"from . import {mod}")

__all__ = funcs + mods
