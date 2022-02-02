import xarray as xr
from glob import glob

ppdir = "/archive/Raphael.Dussin/xanadu_esm4_20190304_mom6_2019.08.08/OM4p25_JRA55do1.4_0netfw_cycle6/gfdl.ncrc4-intel16-prod/pp/ocean_annual/ts/annual"

files = glob(f"{ppdir}/5yr/*tauuo.nc")
files += glob(f"{ppdir}/5yr/*tauvo.nc")
files += glob(f"{ppdir}/1yr/*tauuo.nc")
files += glob(f"{ppdir}/1yr/*tauvo.nc")

ds = xr.open_mfdataset(files, chunks={"time": 5})

ds.sel(time=slice("1999", "2018")).mean(dim="time").to_netcdf(
    "/archive/Raphael.Dussin/datasets/wind_stress_omip2/om4/windstress_OM4p25_JRA1.4_cycle6_1999-2018.nc"
)
ds.sel(time=slice("1959", "1978")).mean(dim="time").to_netcdf(
    "/archive/Raphael.Dussin/datasets/wind_stress_omip2/om4/windstress_OM4p25_JRA1.4_cycle6_1959-1978.nc"
)

