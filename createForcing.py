import numpy as np
import xarray as xr

SPY = 3600*24*365.25

def save_forcing(forcings,fn):
    '''
    Function to save forcings
    '''
    ftime = forcings['time']
    ds = xr.Dataset(
        data_vars=dict(
            Tskin     = (["time"],forcings['Tskin'], {"units":'K'}),
            snowfall  = (["time"],forcings['snowfall'], {"units":'K'}),
            ),
        coords = dict(
        time = ftime,
        ))
    ds['dt']            = forcings['dt']
    ds['T_mean']        = forcings['T_mean']
    ds['snowfall_mean'] = forcings['snowfall_mean']
    ds['rhos']          = forcings['rhos']
    
    ds.to_netcdf(fn)
    ds.close()
### end save_forcing ###
########################

if __name__ == '__main__':
    ftime = np.arange(0,14600)
    fdt = 3600 * 24
    
    fsf = 0.25 # snowfall, m w.e. per year
    sf_std = 0.1
    fsnowfall_we = np.random.normal(fsf,sf_std,len(ftime))
    fsnowfall_we[fsnowfall_we<0] = 0
    fsnowfall = fsnowfall_we * 1000 / SPY # snowfall rate in kg/m2/s

    fts = 250
    ts_std = 15
    fTskin = np.random.normal(fts,ts_std,len(ftime))
    fTskin[fTskin>273] = 273

    rhos = 315
    forcings = {'time':ftime,'dt':fdt,'Tskin':fTskin,'snowfall':fsnowfall,'T_mean':fts,'snowfall_mean':np.mean(fsnowfall),'rhos':rhos}
    filename = 'inputs/miniCFM_forcing_example.nc'
    save_forcing(forcings,filename)