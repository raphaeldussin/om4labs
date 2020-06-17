import numpy as np
import scipy
import xarray as xr
import xesmf as xe


def add_matrix_NaNs(regridder):
    """Helper function to set masked points to NaN instead of zero"""
    X = regridder.weights
    M = scipy.sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.weights = scipy.sparse.coo_matrix(M)
    return regridder


def curv_to_curv(src, dst, reuse_weights=False):
    regridder = xe.Regridder(src, dst, "bilinear", reuse_weights=reuse_weights)
    regridder = add_matrix_NaNs(regridder)
    return regridder(src)
