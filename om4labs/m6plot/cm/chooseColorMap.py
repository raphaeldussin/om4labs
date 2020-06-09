def chooseColorMap(sMin, sMax, difference=None):
    """
  Based on the min/max extremes of the data, choose a colormap that fits the data.
  """
    if difference == True:
        return "dunnePM"
    elif sMin < 0 and sMax > 0:
        return "dunnePM"
    # elif sMax>0 and sMin<0.1*sMax: return 'hot'
    # elif sMin<0 and sMax>0.1*sMin: return 'hot_r'
    else:
        return "dunneRainbow"
