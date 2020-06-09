def regionalMasking(field, yCoord, xCoord, latRange, lonRange):
    maskedField = field.copy()
    maskedField = numpy.ma.masked_where(
        numpy.less(yCoord[0:-1, 0:-1], latRange[0]), maskedField
    )
    maskedField = numpy.ma.masked_where(
        numpy.greater(yCoord[1::, 1::], latRange[1]), maskedField
    )
    maskedField = numpy.ma.masked_where(
        numpy.less(xCoord[0:-1, 0:-1], lonRange[0]), maskedField
    )
    maskedField = numpy.ma.masked_where(
        numpy.greater(xCoord[1::, 1::], lonRange[1]), maskedField
    )
    return maskedField
