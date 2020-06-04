def sectorRanges(sector=None):
  # Should add definitions for tropInd, tropAtl, sAtl, nPac, allPac, allInd, and allAtlantic
  if sector   == 'nAtl':    lonRange=(-100.,40.); latRange=(-15.,80.); hspace=0.25; titleOffset=1.14
  elif sector == 'gomex':   lonRange=(-100.,-50.); latRange=(5.,35.); hspace=0.25; titleOffset=1.14
  elif sector == 'tropPac': lonRange=(-270.,-75.); latRange=(-30.,30.); hspace=-0.2; titleOffset=1.17
  elif sector == 'arctic':  lonRange=(-300,60); latRange=(60.,90.); hspace=0.25; titleOffset=1.14
  elif sector == 'shACC':   lonRange=(-300,60); latRange=(-90,-45.); hspace=0.25; titleOffset=1.14
  else: lonRange=None; latRange=None; hspace=0.25; titleOffset=1.14
  return lonRange, latRange, hspace, titleOffset
