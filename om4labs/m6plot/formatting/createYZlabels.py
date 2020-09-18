def createYZlabels(y, z, ylabel, yunits, zlabel, zunits):
    """
    Checks that y and z labels are appropriate and tries to make some if they are not.
    """
    if y is None:
        if ylabel is None:
            ylabel = "j"
        if yunits is None:
            yunits = ""
    else:
        if ylabel is None:
            ylabel = "Latitude"
        # if yunits is None: yunits=u'\u00B0N'
        if yunits is None:
            yunits = r"$\degree$N"
    if z is None:
        if zlabel is None:
            zlabel = "k"
        if zunits is None:
            zunits = ""
    else:
        if zlabel is None:
            zlabel = "Elevation"
        if zunits is None:
            zunits = "m"
    return ylabel, yunits, zlabel, zunits
