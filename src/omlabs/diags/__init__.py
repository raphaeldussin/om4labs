import os
import warnings

_diag_dir = os.path.split(os.path.abspath(__file__))[0]
_exclude_list = ["__pycache__"]

_diaglist = [_x for _x in os.listdir(_diag_dir)]
_diaglist = [f"{_diag_dir}/{_x}" for _x in _diaglist]
_diaglist = [_x for _x in _diaglist if os.path.isdir(_x) and os.path.basename(_x) not in _exclude_list]
_diaglist = sorted([os.path.basename(_x) for _x in _diaglist])

for _diag in _diaglist:
    try:
        exec(f"from . import {_diag}")
    except Exception as exc:
        warnings.warn(f"Unable to import diagnostic: {_diag}")

del os
del warnings
del _diag
del _diaglist
del _diag_dir
del _exclude_list