import importlib.metadata as ilm

msg = ilm.metadata("omlabs")

__name__ = msg["Name"]
__version__ = msg["Version"]
__license__ = msg["License"]
__email__ = msg["Maintainer-email"]
__description__ = msg["Summary"]
__requires__ = msg["Requires-Dist"]
__requires_python__ = msg["Requires-Python"]

from . import diags
from . import m6plot
from . import om4parser
from . import m6toolbox
