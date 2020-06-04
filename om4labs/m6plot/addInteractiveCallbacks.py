def addInteractiveCallbacks():
  """
  Adds interactive features to a plot on screen.
  Key 'q' to close window.
  Zoom button to center.
  Zoom wheel to zoom in and out.
  """
  def keyPress(event):
    if event.key=='Q': exit(0) # Exit python
    elif event.key=='q': plt.close() # Close just the active figure
  class hiddenStore:
    def __init__(self,axis):
      self.axis = axis
      self.xMin, self.xMax = axis.get_xlim()
      self.yMin, self.yMax = axis.get_ylim()
  save = hiddenStore(plt.gca())
  def zoom(event): # Scroll wheel up/down
    if event.button == 'up': scaleFactor = 1/1.5 # deal with zoom in
    elif event.button == 'down': scaleFactor = 1.5 # deal with zoom out
    elif event.button == 2: scaleFactor = 1.0
    else: return
    axis = event.inaxes
    axmin,axmax=axis.get_xlim(); aymin,aymax=axis.get_ylim();
    (axmin,axmax),(aymin,aymax) = newLims(
        (axmin,axmax), (aymin,aymax), (event.xdata, event.ydata),
        (save.xMin,save.xMax), (save.yMin,save.yMax), scaleFactor)
    if axmin is None: return
    for axis in plt.gcf().get_axes():
      if axis.get_navigate():
        axis.set_xlim(axmin, axmax); axis.set_ylim(aymin, aymax)
    plt.draw() # force re-draw
  def zoom2(event): zoom(event)
  plt.gcf().canvas.mpl_connect('key_press_event', keyPress)
  plt.gcf().canvas.mpl_connect('scroll_event', zoom)
  plt.gcf().canvas.mpl_connect('button_press_event', zoom2)
