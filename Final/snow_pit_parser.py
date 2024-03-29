# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import xarray as xr
from bs4 import BeautifulSoup
import numpy as np
import os

# Grab all the file names in our 
fns = ['./snow_pit_obs/'+file for file in os.listdir('./snow_pit_obs')]

def snowpit_xml_parser(fn):
    """This function will parse a SnowPilot snow pit to format snowpits into an xarray dataset

    Args:
        fn (string): filename

    Returns:
        temps (pd.Series): 
        density (pd.Series):
        pit_time (datetime):
        pit_name (str):
        depths_T (pd.Series):
        obs_dict (dict):
        user_dict (dict):
        layer_dict (dict):
    """
    # Open file
    with open(fn,"r") as file:
        a = BeautifulSoup(file, 'xml')
    # Grab pit observation information for attributes
    obs_dict = a.find('Pit_Observation').attrs
    obs_dict['datetime'] = dt.datetime.fromtimestamp(int(obs_dict['timestamp'])/1000)
    obs_dict['datetime_str'] = obs_dict['datetime'].strftime("%Y-%m/-%d, %H:%M:%S")
    obs_dict['timezone'] = 'MST (UTC-7)'
    obs_dict['elv'] = a.find('Location').attrs['elv']
    # dimensions for xarray dataset
    pit_name = a.find('Location').attrs['name']
    pit_time = obs_dict['datetime']
    pit_depth = float(obs_dict['heightOfSnowpack'])

    # Dictionary of meta data for snowpit observations
    obs_dict = {k: obs_dict[k] for k in ['datetime_str','depthUnits','coordType','elvUnits','pitNotes', 'precip', 'heightOfSnowpack', 'sky', 'aspect',
                                            'windspeed', 
                                            'winDir', 'longitude', 'lat', 'rhoUnits', 'hardnessScaling',
                                            'range', 'state' ]}
                                        
    # User dictionary for snowpit observations
    tmp_dict = a.find('User').attrs
    user_dict = {k: tmp_dict[k] for k in ['measureFrom', 'depthUnits', 'tempUnits', 'coordType', 'elvUnits',
                                            'username', 'useSymbols', 'first', 'last', 'name',
                                            'email', 'affil' ]}   
    # Create series of temperature profile and temperature depths
    tmp_profile = a.find('Temperature_Profile').attrs['temp_profile']
    depths_T = [float(j[0]) for j in [i.split(":") for i in tmp_profile.split(';')]]
    temps = [float(j[1]) for j in [i.split(":") for i in tmp_profile.split(';')]]
    temps_series = pd.Series(temps,index=np.full(len(temps), pit_name))
    # Create series of density profile and density measurement depths
    density_profile = a.find('Density_Profile').attrs['profile']
    depths_rho = [float(j[0]) for j in [i.split(":") for i in density_profile.split(';')]]
    density = [float(j[1]) for j in [i.split(":") for i in density_profile.split(';')]]
    # equate lengths of these series for concatenation 
    for i in range(10):
        if len(density) != len(depths_T):
            density.append(np.nan)
        else:
            break          
    density_series = pd.Series(density,index=np.full(len(density), pit_name))

    layer_dict = {}
    for i,layer in enumerate(a.find_all('Layer')):
        tmp = layer.attrs
        start = tmp['startDepth']
        end = tmp['endDepth']
        if i == 0:
            layer_dict[f'Surface_to_{end}cm'] = tmp
        else:
            layer_dict[f'{start}cm_to_{end}cm'] = tmp
    return temps, density, pit_time, pit_name, depths_T,obs_dict,user_dict, layer_dict

def build_snowpit_ds(fn):
    """Using output from snowpit_xml_parser, build the snowpit dataset for depth, density and temperature

    Args:
        fn (string): filename of snow pilot xml file

    Returns:
        ds: xarray dataset with temperature and density information
    """
    temps, density, pit_time, pit_name, depths_T,obs_dict, user_dict, _ = snowpit_xml_parser(fn) 

    ds = xr.Dataset(
    data_vars=dict(
        temperature=(['depth'],temps,dict(
                                        units='degC',
                                        long_name='Snow pit profile temperature from Cooper-Atkins DFP450W-0-8 4 digital thermometer',
                                        )
                ),
        density=(['depth'],density,dict(
                                        units='kg/m^3',
                                        long_name='Snow density from Snowmetrics density cutter and scale',
                                        measurement_notes='Measurements taken from the surface every 10 cm, so density is a 10-cm layer average.'
                                        )
                )
    ),
    coords=dict(
        depth=depths_T,
        time=pit_time,
        id=pit_name
    ),
    attrs=obs_dict | user_dict
    )
    return ds

def single_layer_ds(fn):
    """Create dataset to hold snow pit layer information

    Args:
        fn (filename): filename for snow pilot xml file

    Returns:
        ds: xarray dataset with information from each layer in the snow pit
    """
    _, _, pit_time, pit_name, _,_,_, layer_dict = snowpit_xml_parser(fn) 
    ds_list = []
    for k in layer_dict.keys():
        ds_list.append(xr.Dataset(layer_dict[k],coords=dict(layer=k,
                                        time=pit_time)))
    ds = xr.concat(ds_list,dim='time')
    ds = ds.where(ds!='', np.nan)

    ds['grainSize'] =ds['grainSize'].astype(float)
    ds['grainSize'].attrs = dict(units='mm') 
    ds['grainSize1'] =ds['grainSize1'].astype(float) 
    ds['grainSize1'].attrs = dict(units='mm')
    ds['startDepth'] =ds['startDepth'].astype(float) 
    ds['startDepth'].attrs = dict(units='cm')
    ds['endDepth'] =ds['endDepth'].astype(float) 
    ds['endDepth'].attrs = dict(units='cm')

    ds = ds[['grainSize',
            'grainSize1',
            'grainType',
            'grainType1',
            'hardness1',
            'hardness2',
            'layerNumber',
            'startDepth',
            'endDepth']]
    return ds

def multi_pit_ds(filenames):
    """Takes a list of xml snow pilot files and concatenates the results into two datasets, 
       one for temperature/density profiles and the other for layer information

    Args:
        filenames (list): list of snow pilot xml file names

    Returns:
        data_ds: xarray dataset with temperature and density information
        layer_ds: xarray dataset with information from each layer in the snow pit
    """
    layer_ds_list = []
    data_ds_list = []
    # Iterate through filenames
    for fn in filenames:
        print(f'Working on {fn}')
        layer_ds_list.append(single_layer_ds(fn))
        data_ds_list.append(build_snowpit_ds(fn))
    print('Done... Now contenating files...')
    # concatenate both datasets
    layer_ds = xr.concat(layer_ds_list, dim='time')
    data_ds = xr.concat(data_ds_list, dim='time')
    print('Finished!')
    return data_ds, layer_ds
ds_data, ds_layer = multi_pit_ds(fns)

# add surface temperatures 
pit_surf_temp = pd.read_csv('./snow_pit_surface_temps.csv', index_col=0)
pit_surf_temp.index = ds_data.time.values
# convert to xarray dataset
ds_surf_temp = xr.Dataset.from_dataframe(pit_surf_temp)
# # rename dimension id
ds_surf_temp = ds_surf_temp.rename({'index': 'time',
                                    'IR':'ir_surface_temp',
                                    'Therm':'thermometer_surface_temp'})
# add a new variable called snow_depth and make in equal to the last depth value in the snowpit and associate it with time
snow_depth = pd.DataFrame([ds_data.isel(time=i).temperature.dropna(dim='depth').depth.max().values for i in range(len(ds_data.time))], index=ds_data.time.values, )
# convert to xarray dataset
snow_depth_ds = xr.Dataset.from_dataframe(snow_depth)
# # rename dimension id
ds_snow_depth = snow_depth_ds.rename({'index': 'time',
                                     0: 'pit_total_snow_depth'})
# merge into ds_data
ds_data = xr.merge([ds_data, ds_surf_temp,ds_snow_depth])


# iterate to set the first value in the temperature variable to thermometer_surface_temp
for i, t in enumerate(ds_data.time.values):
    ds_data.sel(time=t)['temperature'].dropna(dim='depth').values[-1] = ds_data.sel(time=t)['thermometer_surface_temp'].values
# add attribute to each ir_surface_temp
ds_data['ir_surface_temp'].attrs['units'] = 'degC'
ds_data['ir_surface_temp'].attrs['long_name'] = 'Infrared surface temperature from IR thermometer gun'

# add attribute to each thermometer_surface_temp
ds_data['thermometer_surface_temp'].attrs['units'] = 'degC'
ds_data['thermometer_surface_temp'].attrs['long_name'] = 'Surface temperature from Cooper-Atkins DFP450W-0-8 4 digital thermometer'

# add attribute to each pit_total_snow_depth
ds_data['pit_total_snow_depth'].attrs['units'] = 'cm'
ds_data['pit_total_snow_depth'].attrs['long_name'] = 'Total snow depth from snow pit'

# convert time to utc
ds_data["time"] = pd.to_datetime(ds_data.time.values).tz_localize('MST').tz_convert('UTC').astype('datetime64[ns]')
# rename datetime_str attribute to local_time_MST
ds_data.attrs['local_time'] = 'MST (UTC-7)'
# add attribute about time
ds_data.attrs['time_info'] = 'Start time of snow pit measurements (beginning with surface temperature)'
ds_data.attrs['timezone_info'] = 'Data presented in UTC'

# Save to netcdf
if __name__ == "__main__":
    ds_data.to_netcdf("./kettle_ponds_snowpit_profiles.nc")
    ds_layer.to_netcdf("./kettle_ponds_snowpit_layers.nc")
if not os.path.exists("./kettle_ponds_snowpit_profiles.nc"):
    ds_data.to_netcdf("./kettle_ponds_snowpit_profiles.nc")
    ds_layer.to_netcdf("./kettle_ponds_snowpit_layers.nc")