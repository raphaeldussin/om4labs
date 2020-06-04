def dunne_pm(N=256):
  """
  Plus-minus  colormap from John Dunne.
  """
  cdict = {'red':   [(0.00, 0.3, 0.3),
                     (0.05, 0.5, 0.5),
                     (0.20, 0.0, 0.0),
                     (0.30, 0.4, 0.4),
                     (0.40, 0.8, 0.8),
                     (0.50, 1.0, 1.0),
                     (0.95, 0.6, 0.6),
                     (1.00, 0.4, 0.4)],

           'green': [(0.00, 0.0, 0.0),
                     (0.30, 0.5, 0.5),
                     (0.40, 1.0, 1.0),
                     (0.70, 1.0, 1.0),
                     (1.00, 0.0, 0.0)],

           'blue':  [(0.00, 0.3, 0.3),
                     (0.05, 0.5, 0.5),
                     (0.20, 1.0, 1.0),
                     (0.50, 1.0, 1.0),
                     (0.60, 0.7, 0.7),
                     (0.70, 0.0, 0.0),
                     (1.00, 0.0, 0.0)]}
  import matplotlib
  cmap = matplotlib.colors.LinearSegmentedColormap('dunnePM', cdict, N=N)
  cmap.set_under([.1,.0,.1]); cmap.set_over([.2,0.,0.])
  #cmap.set_bad('w')
  matplotlib.cm.register_cmap(cmap=cmap)
  return cmap
