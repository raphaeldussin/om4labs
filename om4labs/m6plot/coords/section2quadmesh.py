import numpy as np

def section2quadmesh(x, z, q, representation='pcm'):
  """
  Creates the appropriate quadmesh coordinates to plot a scalar q(1:nk,1:ni) at
  horizontal positions x(1:ni+1) and between interfaces at z(nk+1,ni), using
  various representations of the topography.
  Returns X(2*ni+1), Z(nk+1,2*ni+1) and Q(nk,2*ni) to be passed to pcolormesh.
  TBD: Optionally, x can be dimensioned as x(ni) in which case it will be extraplated as if it had 
  had dimensions x(ni+1).
  
  Optional argument:
  
  representation='pcm' (default) yields a step-wise visualization, appropriate for
           z-coordinate models.
  representation='plm' yields a piecewise-linear visualization more representative
           of general-coordinate (and isopycnal) models.
  representation='linear' is the aesthetically most pleasing but does not
           represent the data conservatively.
  """

  if x.ndim!=1: raise Exception('The x argument must be a vector')
  if z.ndim!=2: raise Exception('The z argument should be a 2D array')
  if q.ndim!=2: raise Exception('The z argument should be a 2D array')
  qnk, qni = q.shape
  znk, zni = z.shape
  xni = x.size
  if zni!=qni: raise Exception('The last dimension of z and q must be equal in length')
  if znk!=qnk+1: raise Exception('The first dimension of z must be 1 longer than that of q. q has %i levels'%qnk)
  if xni!=qni+1: raise Exception('The length of x must 1 longer than the last dimension of q')

  if type( z ) == np.ma.core.MaskedArray: z[z.mask] = 0
  if type( q ) == np.ma.core.MaskedArray: qmin = np.amin(q); q[q.mask] = qmin

  periodicDomain =  abs((x[-1]-x[0])-360. ) < 1e-6 # Detect if horizontal axis is a periodic domain

  if representation=='pcm':
    X = np.zeros((2*qni))
    X[::2] = x[:-1]
    X[1::2] = x[1:]
    Z = np.zeros((qnk+1,2*qni))
    Z[:,::2] = z
    Z[:,1::2] = z
    Q = np.zeros((qnk,2*qni-1))
    Q[:,::2] = q
    Q[:,1::2] = ( q[:,:-1] + q[:,1:] )/2.
  elif representation=='linear':
    X = np.zeros((2*qni+1))
    X[::2] = x
    X[1::2] = ( x[0:-1] + x[1:] )/2.
    Z = np.zeros((qnk+1,2*qni+1))
    Z[:,1::2] = z
    Z[:,2:-1:2] = ( z[:,0:-1] + z[:,1:] )/2.
    Z[:,0] = z[:,0]
    Z[:,-1] = z[:,-1]
    Q = np.zeros((qnk,2*qni))
    Q[:,::2] = q
    Q[:,1::2] = q
  elif representation=='plm':
    X = np.zeros((2*qni))
    X[::2] = x[:-1]
    X[1::2] = x[1:]
    # PLM reconstruction for Z
    dz = np.roll(z,-1,axis=1) - z # Right-sided difference
    if not periodicDomain: dz[:,-1] = 0 # Non-periodic boundary
    d2 = ( np.roll(z,-1,axis=1) - np.roll(z,1,axis=1) )/2. # Centered difference
    d2 = ( dz + np.roll(dz,1,axis=1) )/2. # Centered difference
    s = np.sign( d2 ) # Sign of centered slope
    s[dz * np.roll(dz,1,axis=1) <= 0] = 0 # Flatten extrema
    dz = np.abs(dz) # Only need magnitude from here on
    S = s * np.minimum( np.abs(d2), np.minimum( dz, np.roll(dz,1,axis=1) ) ) # PLM slope
    Z = np.zeros((qnk+1,2*qni))
    Z[:,::2] = z - S/2.
    Z[:,1::2] = z + S/2.
    Q = np.zeros((qnk,2*qni-1))
    Q[:,::2] = q
    Q[:,1::2] = ( q[:,:-1] + q[:,1:] )/2.
  else: raise Exception('Unknown representation!')

  return X, Z, Q
