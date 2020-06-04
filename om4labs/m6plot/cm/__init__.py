# -*- coding: utf-8 -*-
"""m6plot.cm - routines to handle colormaps used by matplotlib
"""

__version__ = '0.0.0'
__all__ = [
    'chooseColorLevels',
    'chooseColorMap',
    'dunne_pm',
    'dunne_rainbow',
    ]

for func in __all__:
    exec(f'from .{func} import *')
