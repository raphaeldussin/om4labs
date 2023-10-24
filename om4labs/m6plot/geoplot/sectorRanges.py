def sectorRanges(sector=None):
    # Should add definitions for tropInd, tropAtl, sAtl, nPac, allPac, allInd, and allAtlantic
    if sector == "nAtl":
        lonRange = (-100.0, 40.0)
        latRange = (-15.0, 80.0)
        hspace = 0.25
        titleOffset = 1.14
    elif sector == "gomex":
        lonRange = (-100.0, -50.0)
        latRange = (5.0, 35.0)
        hspace = 0.25
        titleOffset = 1.14
    elif sector == "tropPac":
        lonRange = (-270.0, -75.0)
        latRange = (-30.0, 30.0)
        hspace = -0.2
        titleOffset = 1.17
    elif sector == "arctic":
        lonRange = (-300, 60)
        latRange = (60.0, 90.0)
        hspace = 0.25
        titleOffset = 1.14
    elif sector == "shACC":
        lonRange = (-300, 60)
        latRange = (-90, -45.0)
        hspace = 0.25
        titleOffset = 1.14
    else:
        lonRange = None
        latRange = None
        hspace = 0.25
        titleOffset = 1.14
    return lonRange, latRange, hspace, titleOffset
