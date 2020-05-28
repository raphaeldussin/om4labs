import xarray as xr
import numpy as np


grid_1x1 = xr.Dataset()
grid_1x1['lon'] = xr.DataArray(data=np.arange(0.5,360.5), dims=('lon'))
grid_1x1['lat'] = xr.DataArray(data=np.arange(-89.5,90.5), dims=('lat'))
grid_1x1['bnds'] = xr.DataArray(data=[0, 1], dims=('bnds'))
grid_1x1['lon_bnds'] = xr.DataArray(data=[np.arange(0,360),
                                          np.arange(1,361)], dims=('bnds','lon'))
grid_1x1['lat_bnds'] = xr.DataArray(data=[np.arange(-90,90),
                                          np.arange(-89,91)], dims=('bnds','lat'))


def test_infer_and_assign_coord():
    from OM4_Analysis_Labs.om4compute import infer_and_assign_coord
    da = xr.DataArray(data=np.empty((360, 2)), dims=('lon', 'bnds'))
    da = infer_and_assign_coord(grid_1x1, da, 'lon')
    assert "assigned_lon" in list(da.dims)


def test_copy_coordinates():
    from OM4_Analysis_Labs.om4compute import copy_coordinates
    da = xr.DataArray(data=np.empty((360, 2)), dims=('lon', 'bnds'))
    da = copy_coordinates(grid_1x1['lon_bnds'], da, ['lon', 'bnds'])
    assert list(da.coords).sort() == list(grid_1x1['lon_bnds'].coords).sort()


def test_compute_area_regular_grid():
    from OM4_Analysis_Labs.om4compute import compute_area_regular_grid
    area = compute_area_regular_grid(grid_1x1)
    assert isinstance(area, np.ndarray)
    assert area.shape == (180, 360)
