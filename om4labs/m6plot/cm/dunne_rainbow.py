def dunne_rainbow(N=256):
  """
  Spectral/rainbow colormap from John Dunne.
  """
  cdict = {'red': [(0.00, 0.95, 0.95),
                   (0.09, 0.85, 0.85),
                   (0.18, 0.60, 0.60),
                   (0.32, 0.30, 0.30),
                   (0.45, 0.00, 0.00),
                   (0.60, 1.00, 1.00),
                   (0.85, 1.00, 1.00),
                   (1.00, 0.40, 0.00)],

         'green': [(0.00, 0.75, 0.75),
                   (0.09, 0.85, 0.85),
                   (0.18, 0.60, 0.60),
                   (0.32, 0.20, 0.20),
                   (0.45, 0.60, 0.60),
                   (0.60, 1.00, 1.00),
                   (0.73, 0.70, 0.70),
                   (0.85, 0.00, 0.00),
                   (1.00, 0.00, 0.00)],

         'blue':  [(0.00, 1.00, 1.00),
                   (0.32, 1.00, 1.00),
                   (0.45, 0.30, 0.30),
                   (0.60, 0.00, 0.00),
                   (1.00, 0.00, 0.00)]}
  import matplotlib
  cmap = matplotlib.colors.LinearSegmentedColormap('dunneRainbow', cdict, N=N)
  #cmap.set_under([1,.65,.85]); cmap.set_over([.25,0.,0.])
  cmap.set_under([.95*.9,.75*.9,.9]); cmap.set_over([.3,0.,0.])
  #cmap.set_bad('w')
  matplotlib.cm.register_cmap(cmap=cmap)
  return cmap
