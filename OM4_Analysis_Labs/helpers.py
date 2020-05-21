import xarray as xr

""" A set of helper functions """


def try_variable_from_list(vars_in_file, query_vars):
    """test variables from query_vars to find which one
       is present in vars_in_file """
    for var in query_vars:
        if var in vars_in_file:
            return var
    return None


def get_run_name(ncfile):
    ds = xr.open_mfdataset(ncfile, combine='by_coords')
    if 'title' in ds.attrs:
        return ds.attrs['title']
    else:
        return 'Unknown experiment'
