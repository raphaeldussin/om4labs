def label(label, units):
    """
  Combines a label string and units string together in the form 'label [units]'
  unless one of the other is empty.
  """
    string = r"" + label
    if len(units) > 0:
        string = string + " [" + units + "]"
    return string
