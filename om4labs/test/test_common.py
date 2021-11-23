import xarray as xr
import numpy as np


grid_1x1 = xr.Dataset()
grid_1x1["lon"] = xr.DataArray(data=np.arange(0.5, 360.5), dims=("lon"))
grid_1x1["lat"] = xr.DataArray(data=np.arange(-89.5, 90.5), dims=("lat"))
grid_1x1["bnds"] = xr.DataArray(data=[0, 1], dims=("bnds"))
grid_1x1["lon_bnds"] = xr.DataArray(
    data=[np.arange(0, 360), np.arange(1, 361)], dims=("bnds", "lon")
)
grid_1x1["lat_bnds"] = xr.DataArray(
    data=[np.arange(-90, 90), np.arange(-89, 91)], dims=("bnds", "lat")
)


def test_infer_and_assign_coord():
    from om4labs.om4common import infer_and_assign_coord

    da = xr.DataArray(data=np.empty((360, 2)), dims=("lon", "bnds"))
    da = infer_and_assign_coord(grid_1x1, da, "lon")
    assert "assigned_lon" in list(da.dims)


def test_copy_coordinates():
    from om4labs.om4common import copy_coordinates

    da = xr.DataArray(data=np.empty((360, 2)), dims=("lon", "bnds"))
    da = copy_coordinates(grid_1x1["lon_bnds"], da, ["lon", "bnds"])
    assert list(da.coords).sort() == list(grid_1x1["lon_bnds"].coords).sort()


def test_compute_area_regular_grid():
    from om4labs.om4common import compute_area_regular_grid

    area = compute_area_regular_grid(grid_1x1)
    assert isinstance(area, np.ndarray)
    assert area.shape == (180, 360)


def test_standardize_longitude():
    from om4labs.om4common import standardize_longitude

    # case 1: extending past 360.
    lon = np.arange(20.5, 380.5, 1.0)
    lat = np.arange(-89.5, 90.5, 1)
    data = np.random.normal(size=(len(lat), len(lon)))
    data = xr.DataArray(
        data, dims={"lat": lat, "lon": lon}, coords={"lat": lat, "lon": lon}
    )
    dset = xr.Dataset({"data": data})

    # 0 to 360
    result = standardize_longitude(dset, "lon")
    assert np.all(dset.sel(lon=370.5).data.values == result.sel(lon=10.5).data.values)
    assert np.all(dset.sel(lon=60.5).data.values == result.sel(lon=60.5).data.values)

    # -180 to 180
    result = standardize_longitude(dset, "lon", start_lon=-180)
    assert np.all(dset.sel(lon=370.5).data.values == result.sel(lon=10.5).data.values)
    assert np.all(dset.sel(lon=209.5).data.values == result.sel(lon=-150.5).data.values)

    # case 2: arbitrary start
    lon = np.arange(-159.5, 200.5, 1.0)
    lat = np.arange(-89.5, 90.5, 1)
    data = np.random.normal(size=(len(lat), len(lon)))
    data = xr.DataArray(
        data, dims={"lat": lat, "lon": lon}, coords={"lat": lat, "lon": lon}
    )
    dset = xr.Dataset({"data": data})

    # 0 to 360
    result = standardize_longitude(dset, "lon")
    assert np.all(dset.sel(lon=-140.5).data.values == result.sel(lon=219.5).data.values)
    assert np.all(dset.sel(lon=60.5).data.values == result.sel(lon=60.5).data.values)

    # -180 to 180
    result = standardize_longitude(dset, "lon", start_lon=-180)
    assert np.all(dset.sel(lon=190.5).data.values == result.sel(lon=-169.5).data.values)
    assert np.all(dset.sel(lon=-60.5).data.values == result.sel(lon=-60.5).data.values)

    # case 3: 2-dimensional coordinates
    x = np.arange(1, 361, 1)
    y = np.arange(1, 181, 1)
    lon = np.arange(-159.5, 200.5, 1.0)
    lat = np.arange(-89.5, 90.5, 1)
    geolon, geolat = np.meshgrid(lon, lat)
    data = np.random.normal(size=geolon.shape)
    geolat = xr.DataArray(geolat, dims={"y": y, "x": x}, coords={"y": y, "x": x})
    geolon = xr.DataArray(geolon, dims={"y": y, "x": x}, coords={"y": y, "x": x})
    data = xr.DataArray(
        data, dims={"y": y, "x": x}, coords={"geolat": geolat, "geolon": geolon}
    )
    dset = xr.Dataset({"data": data})
    result = standardize_longitude(dset, "geolon")
    assert (result.geolon.min() > 0) & (result.geolon.max() < 360)
