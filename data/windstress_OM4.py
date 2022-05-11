import xarray as xr
from glob import glob
from xgcm import Grid
import xesmf
import subprocess as sp


# outputdir = "/archive/Raphael.Dussin/datasets/wind_stress_omip2/om4"
outputdir = "."


ppdir_OM4p25 = "/archive/Raphael.Dussin/xanadu_esm4_20190304_mom6_2019.08.08/OM4p25_JRA55do1.4_0netfw_cycle6/gfdl.ncrc4-intel16-prod/pp/ocean_annual/"

ppdir_OM4p125 = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20210706/CM4_piControl_c192_OM4p125_v6_alt1/gfdl.ncrc4-intel18-prod-openmp/pp/ocean_annual"

# Read the files from the 1/4 degree
files = glob(f"{ppdir_OM4p25}/ts/annual/5yr/*tauuo.nc")
files += glob(f"{ppdir_OM4p25}/ts/annual/5yr/*tauvo.nc")
files += glob(f"{ppdir_OM4p25}/ts/annual/1yr/*tauuo.nc")
files += glob(f"{ppdir_OM4p25}/ts/annual/1yr/*tauvo.nc")
files += glob(f"{ppdir_OM4p25}/ocean_annual.static.nc")

sp.check_call(
    "dmget /archive/Raphael.Dussin/xanadu_esm4_20190304_mom6_2019.08.08/OM4p25_JRA55do1.4_0netfw_cycle6/gfdl.ncrc4-intel16-prod/pp/ocean_annual/ts/annual/5yr/*tauuo.nc /archive/Raphael.Dussin/xanadu_esm4_20190304_mom6_2019.08.08/OM4p25_JRA55do1.4_0netfw_cycle6/gfdl.ncrc4-intel16-prod/pp/ocean_annual/ts/annual/5yr/*tauvo.nc /archive/Raphael.Dussin/xanadu_esm4_20190304_mom6_2019.08.08/OM4p25_JRA55do1.4_0netfw_cycle6/gfdl.ncrc4-intel16-prod/pp/ocean_annual/ts/annual/1yr/*tauuo.nc /archive/Raphael.Dussin/xanadu_esm4_20190304_mom6_2019.08.08/OM4p25_JRA55do1.4_0netfw_cycle6/gfdl.ncrc4-intel16-prod/pp/ocean_annual/ts/annual/1yr/*tauvo.nc /archive/Raphael.Dussin/xanadu_esm4_20190304_mom6_2019.08.08/OM4p25_JRA55do1.4_0netfw_cycle6/gfdl.ncrc4-intel16-prod/pp/ocean_annual/ocean_annual.static.nc",
    shell=True,
)

ds = xr.open_mfdataset(files, chunks={"time": 5})

# build xgcm grid
grid = Grid(
    ds,
    coords={
        "X": {"center": "xh", "right": "xq"},
        "Y": {"center": "yh", "right": "yq"},
    },
    periodic=["X"],
)

# build wind stress curl
stress_curl = -grid.diff(
    ds["tauuo"].fillna(0.0) * ds["dxCu"], "Y", boundary="fill"
) + grid.diff(ds["tauvo"].fillna(0.0) * ds["dyCv"], "X", boundary="fill")

stress_curl = stress_curl / (ds["areacello_bu"] * 1035.0)
stress_curl = stress_curl.where(ds["wet_c"] == 1)
stress_curl = stress_curl.assign_coords({"lon": ds["geolon_c"], "lat": ds["geolon_c"]})

# write means on the 1/4 degree grid
stress_curl_1999_2018 = (
    stress_curl.sel(time=slice("1999", "2018")).mean(dim="time").to_dataset(name="curl")
)
stress_curl_1959_1978 = (
    stress_curl.sel(time=slice("1959", "1978")).mean(dim="time").to_dataset(name="curl")
)


stress_curl_1999_2018.rename({"lon": "geolon_c", "lat": "geolat_c"}).to_netcdf(
    f"{outputdir}/windstresscurl_OM4p25_JRA1.4_cycle6_1999-2018.nc",
    format="NETCDF3_64BIT",
)
stress_curl_1959_1978.rename({"lon": "geolon_c", "lat": "geolat_c"}).to_netcdf(
    f"{outputdir}/windstresscurl_OM4p25_JRA1.4_cycle6_1959_1978.nc",
    format="NETCDF3_64BIT",
)


# prepare grids for remapping
prepare_grids = False

if prepare_grids:
    # OM4p25
    grid_OM4p25 = xr.open_dataset(f"{ppdir_OM4p25}/ocean_annual.static.nc")
    grid_src = xr.Dataset()
    grid_src["lon"] = grid_OM4p25["geolon_c"]
    grid_src["lat"] = grid_OM4p25["geolat_c"]

    # OM4p125
    grid_OM4p125 = xr.open_dataset(f"{ppdir_OM4p125}/ocean_annual.static.nc")
    grid_dst = xr.Dataset()
    grid_dst["lon"] = grid_OM4p125["geolon_c"]
    grid_dst["lat"] = grid_OM4p125["geolat_c"]

    grid_src.drop_vars(["xq", "yq"]).to_netcdf(f"{outputdir}/grid_cm4p25_corners.nc")
    grid_dst.drop_vars(["xq", "yq"]).to_netcdf(f"{outputdir}/grid_cm4p125_corners.nc")
else:
    grid_src = xr.open_dataset(f"{outputdir}/grid_cm4p25_corners.nc")
    grid_dst = xr.open_dataset(f"{outputdir}/grid_cm4p125_corners.nc")


# Create weights with command line tool... sigh...
sp.check_call(
    f"ESMF_RegridWeightGen -s {outputdir}/grid_cm4p25_corners.nc -d {outputdir}/grid_cm4p125_corners.nc -m neareststod -w {outputdir}/remap_wgts_nn_p25_to_p125.nc",
    shell=True,
)


# Remap using the pre-computed weights
remap = xesmf.Regridder(
    grid_src,
    grid_dst,
    "nearest_s2d",
    periodic=True,
    reuse_weights=True,
    weights=f"{outputdir}/remap_wgts_nn_p25_to_p125.nc",
)

stress_curl_1999_2018_remapped = remap(stress_curl_1999_2018.chunk(dict(xq=-1, yq=-1)))
stress_curl_1959_1978_remapped = remap(stress_curl_1959_1978.chunk(dict(xq=-1, yq=-1)))

stress_curl_1999_2018_remapped.rename({"lon": "geolon_c", "lat": "geolat_c"}).to_netcdf(
    f"{outputdir}/windstresscurl_OM4p25_JRA1.4_cycle6_1999-2018_remapped_p125.nc",
    format="NETCDF3_64BIT",
)
stress_curl_1959_1978_remapped.rename({"lon": "geolon_c", "lat": "geolat_c"}).to_netcdf(
    f"{outputdir}/windstresscurl_OM4p25_JRA1.4_cycle6_1959_1978_remapped_p125.nc",
    format="NETCDF3_64BIT",
)
