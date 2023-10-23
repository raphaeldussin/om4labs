""" omlabs lexicon for default diagnostic CLI options """

import argparse


class DefaultDictParser(argparse.ArgumentParser):
    """Custom argparse class extension capable of returning a dictionary
    object. This class will bypass the error when called without
    required options and will prepoulate with default values

    Parameters
    ----------
    argparse : argparse.ArgumentParser

    See Also
    --------
    default_diag_parser : function to return default options
    """

    def error(self, message):
        actions = self.__dict__["_actions"]
        defaults = {}
        for act in actions[1::]:
            defaults[act.__dict__["dest"]] = act.__dict__["default"]
        return defaults


def default_diag_parser(description="", template=False, exclude=None):
    """Establishes a default ArgumentParser object with default values.
    Options may be omitted if desired. The default set includes:

    Baseline options:

    These options control common framework functions:

    INFILE:  path; dataset, list of datasets, or regular expression
    pointing to input files

    -l, --label:        str; string to use in plot titles
                        (typically the experiment name)
    -o, --outdir:       path; directory to store plots.  Will create if not present
    -d, --dpi:          int; figure resolution in dots per inch (default is 100)
    -i, --interactive:  bool; displays figures to screen if present
    -v, --verbose:      turn on verbose output.  Need to consider multiple
                        levels of verbosity
    -S, --suptitle:     str; super-title to annotate figures

    Grid options:

    The horizontal and vertical grid information is contained in the gridSpec tar file
    and will be the preferred route for obtaining this data.  Some grids, such as the
    1x1 spherical grid, do not have a gridSpec tar file.  In this case, an
    ocean.static dataset may be provided

    -g, --gridspec:     path;  gridspec tarfile
    -s, --static:       path;  ocean.static dataset

    Intake catalogs:

    Intake catalogs will be used to specify default grid files for known model
    configurations and for paths to observational datasets.  These will be the
    preferred paths:

    -c, --config:       str; model configuration (e.g. OM4p25, OM4p5, OM4p125)
    -p, --platform:     str; name of system. Used to find intake catalogs
                        (e.g. gfdl, orion, gaea)

    Overrides:

    Options are available to override the intake catalog values.  Overrides will
    be defined in long format, e.g. no short single letter options:

    --obsfile:      path; observational file
    --topog:        path; topography file
    --hgrid:        path; ocean hgrid file
    --basin:        path; file containing basin codes


    Parameters
    ----------
    description : str, optional
        descriptive string to pass to the CLI, by default ""
    template : bool, optional
        return a DefaultDictParser object if True, by default False
    exclude : str, list, optional
        str or list of str of options to exclude, by default None

    Returns
    -------
    ArgumentParser
        pre-populated ArgumentParser object with default opts

    See Also
    --------
    DefaultDictParser : custom class capable of returning a dict() of CLI args

    """
    if exclude is not None:
        if not isinstance(exclude, list):
            exclude = [exclude]
    else:
        exclude = []

    if template is True:
        parser = DefaultDictParser(
            description=description, formatter_class=argparse.RawTextHelpFormatter
        )
    else:
        parser = argparse.ArgumentParser(
            description=description, formatter_class=argparse.RawTextHelpFormatter
        )

    # baseline options
    if "infile" not in exclude:
        parser.add_argument(
            "infile",
            metavar="INFILE",
            type=str,
            nargs="+",
            help="Path to input NetCDF file(s)",
        )

    if "label" not in exclude:
        parser.add_argument(
            "-l",
            "--label",
            type=str,
            default="",
            help="String label to annotate the plot",
        )

    if "outdir" not in exclude:
        parser.add_argument(
            "-o",
            "--outdir",
            type=str,
            default="./",
            help="Output directory. Default is current directory",
        )

    if "dpi" not in exclude:
        parser.add_argument(
            "-d",
            "--dpi",
            type=int,
            default=100,
            help="Figure resolution in dots per inch. Default is 100.",
        )

    if "interactive" not in exclude:
        parser.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help="Interactive mode displays plot to screen. Default is False",
        )

    if "verbose" not in exclude:
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Verbose output",
        )

    if "suptitle" not in exclude:
        parser.add_argument(
            "-S",
            "--suptitle",
            type=str,
            default="",
            help="Super-title for annotating figures",
        )

    if "format" not in exclude:
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            default="png",
            help="Output format for plots. Default is png",
        )

    # grid-related options
    if "gridspec" not in exclude:
        parser.add_argument(
            "-g", "--gridspec", type=str, default=None, help="Path to gridspec tar file"
        )

    if "static" not in exclude:
        parser.add_argument(
            "-s",
            "--static",
            type=str,
            default=None,
            help="Path to ocean.static.nc file",
        )

    # intake catalog-related options
    if "config" not in exclude:
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            help="Model configuration, default is OM4",
        )

    if "platform" not in exclude:
        parser.add_argument(
            "--platform",
            type=str,
            required=False,
            default="gfdl",
            help="Computing platform, default is gfdl",
        )

    if "basin" not in exclude:
        # override arguments
        parser.add_argument(
            "--basin",
            type=str,
            default=None,
            help="Path to basin code file",
        )

    if "obsfile" not in exclude:
        parser.add_argument(
            "--obsfile",
            type=str,
            default=None,
            help="Path to observational dataset file",
        )

    if "hgrid" not in exclude:
        parser.add_argument(
            "--hgrid",
            type=str,
            default=None,
            help="Path to hgrid file",
        )

    if "topog" not in exclude:
        parser.add_argument(
            "--topog",
            type=str,
            default=None,
            help="Path to topography file",
        )

    return parser
