import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import xarray as xr
from xgcm import Grid

def parse():
    return

def read(dictArgs):
    ds        = xr.open_mfdataset(dictArgs["infile"])
    ds_static = xr.open_mfdataset(dictArgs["static"])
    return ds, ds_static


def calculate(
    ds, ds_static, varx="tauuo", vary="tauvo", areacello_bu="areacello_bu", xdim="lon", ydim="lat"
               ):
    """Calculate curl of stress acting on surface of the ocean 

    Calculation function
    Parameters
        ----------
    ds        : xarray.Dataset dataset with tauuo and tauvo
    ds_static : xarray.Dataset with grid values
    varname   : str, optional
        Name of the tauuo and tauvo variables, by default "tauuo" and "tauvo"
    area : str, optional
        Name of the area variable, by default "areacello"
    xdim : str, optional
        Name of the longitude coordinate, by default "lon"
    ydim : str, optional
        Name of the latitude coordinate, by default "lat"
    Returns
    -------
        xarray.DataArray stress_curl 
        curl of surface ocean stress
    """

    rho0 = 1035.0
    area = ds_static[areacello_bu]
    taux = ds_modeldata[varx]
    tauy = ds_modeldata[vary]

    stress_curl = ( - grid.diff(taux * ds_static.dxCu, 'Y', boundary='fill')
                    + grid.diff(tauy * ds_static.dyCv, 'X', boundary='fill') )

    stress_curl = stress_curl/(area*rho0)
    stress_curl = stress_curl.where(ds_static["wet_c"]==1)
    stress_curl = stress_curl.assign_coords({'geolon_c': ds_static['geolon_c'], 'geolat_c': ds_static['geolat_c']})

    return stress_curl


def plot(field, vmin=-3e-10, vmax=3e-10, lat_lon_ext = [-180, 180, -85., 90.],
         lon='geolon', lat='geolat', cmap=cmocean.cm.delta, title='stress curl'):

    fig = plt.figure(figsize=[22,8])
    ax  = fig.add_subplot(projection=ccrs.Robinson(),facecolor='grey')
    p   = field.plot(ax=ax, x="geolon_c", y="geolat_c", vmin=vmin, vmax=vmax, cmap=cmap,
                     transform=ccrs.PlateCarree(), add_labels=True, add_colorbar=False)

    # add separate colorbar
    cb = plt.colorbar(p, ax=ax, format='%.1e', extend='both', shrink=0.6)
    cb.ax.tick_params(labelsize=12)

    # add gridlines and extent of lat/lon
    p.axes.gridlines(color='black', alpha=0.5, linestyle='--')
    ax.set_extent(lat_lon_ext, crs=ccrs.PlateCarree())
    _ = plt.title(title, fontsize=14)

    return fig
    


def run(dictArgs):
        # set visual backend
        if dictArgs["interactive"] is False:
            plt.switch_backend("Agg")

        ds, ds_static = read(dictArgs)
        darray = calculate(ds,ds_static)
        figs   = plot(darray)
        figs   = [figs] if not isinstance(figs,list) else figs
        assert isinstance(figs,list), "Figures must be inside a list object"

        filenames = [
            f"{dictArgs['outdir']}/Curl of surface stress (N/m^3)",
        ]

        imgbufs = image_handler(figs, dictArgs, filename=filenames)

        return imgbufs
    
        
def run():
    return


def parse_and_run():
    return

