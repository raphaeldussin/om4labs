from om4labs.diags.generic_annual_bias_1x1deg.generic_annual_bias_1x1deg import parse, run
from om4labs.m6plot.formatting import pmCI, linCI

possible_variable_names = ["thetao", "temp", "ptemp", "TEMP", "PTEMP"]


def parse_and_run(cliargs=None):
    cmdLineArgs = parse(cliargs)
    dictArgs = vars(cmdLineArgs)  # convert parser args to dict
    # add custom option for data read
    dictArgs['possible_variable_names'] = possible_variable_names
    dictArgs['surface_default_depth'] = 2.5
    # add custom options for plot
    dictArgs['var'] = "SST"
    dictArgs['units'] = "[$\degree$C]"
    dictArgs['clim_diff'] = pmCI(0.25, 4.5, 0.5)
    dictArgs['clim_compare'] = linCI(-2, 29, 0.5)
    dictArgs['cmap_diff'] = "dunnePM"
    dictArgs['cmap_compare'] = "dunneRainbow"
    # call run from generic
    run(dictArgs)
