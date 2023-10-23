from .. import generic_yz_annual_bias_1x1deg as generic

from omlabs.m6plot.formatting import pmCI, linCI

possible_variable_names = ["so", "salt", "salinity", "SALT", "SALINITY"]


def parse_and_run(cliargs=None):
    dictArgs = parse(cliargs)
    imgbufs = run(dictArgs)
    return imgbufs


def parse(cliargs=None, template=False):
    if template is True:
        dictArgs = generic.parse(template=True)
    else:
        cmdLineArgs = generic.parse(cliargs)
        dictArgs = vars(cmdLineArgs)  # convert parser args to dict
    # add custom option for data read
    dictArgs["possible_variable_names"] = possible_variable_names
    dictArgs["surface_default_depth"] = 2.5
    # add custom options for plot
    dictArgs["var"] = "so"
    dictArgs["units"] = "[ppt]"
    dictArgs["clim_diff"] = pmCI(0.125, 2.25, 0.25)
    dictArgs["clim_compare"] = linCI(20, 30, 10, 31, 39, 0.5)
    dictArgs["cmap_diff"] = "dunnePM"
    dictArgs["cmap_compare"] = "dunneRainbow"
    return dictArgs


def run(dictArgs):
    imgbufs = generic.run(dictArgs)
    return imgbufs
