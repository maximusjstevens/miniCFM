#!/usr/bin/env python
'''
miniCFM.py
======================
miniature version of CFM to be coupled to ISSM.

Copyright © 2023 C. Max Stevens

Distributed under terms of the MIT license. 
'''

import numpy as np
import pandas as pd
import xarray as xr
import os
from scipy.sparse import spdiags
import scipy.sparse.linalg as splin

### global constants
SPY = 3600*24*365.25
GRAVITY = 9.8
R = 8.314
RHO_I_MGM   = 0.917 
RHO_1_MGM   = 0.550
RHO_I = 917
CP_I = 2097.0 # [J/kg/K]

def hl_analytic(dz, rhos0, THL, AHL):
    '''
    Model steady-state firn density profiles and bubble close-off, uses m w.e. a^-1
    full HL in CFM includes age; this is abridged version for density only.
    This is just for generating a spin up/restart

    : rhos0: surface density
                (unit: kg/m3, type: float)
    : h: depths at which to caclulate density
                (unit: m, type: float)
    : THL: mean annual temperature
                (unit: K, type: float)
    : AHL: mean annual accumulation
                (unit: kg/m2/s, type: float)

    '''
    z = np.append(0,np.cumsum(dz))
    h = (z[1:] + z[:-1])/2

    rhos = rhos0 / 1000.0

    A = AHL * SPY / 1000 # put into m w.e. per year
    k0 = 11.0 * np.exp(-10160 / (R * THL))
    k1 = 575.0 * np.exp(-21400 / (R * THL))

    # depth of critical density, eqn 8 from Herron and Langway
    h0_55 = 1 / (RHO_I_MGM * k0) * (np.log(RHO_1_MGM / (RHO_I_MGM - RHO_1_MGM)) - np.log(rhos / (RHO_I_MGM - rhos)))
    Z0 = np.exp(RHO_I_MGM * k0 * h + np.log(rhos / (RHO_I_MGM - rhos)))

    Z1 = np.exp(RHO_I_MGM * k1 * (h - h0_55) / np.sqrt(A) + np.log(RHO_1_MGM / (RHO_I_MGM - RHO_1_MGM)))
    Z = np.concatenate((Z0[h < h0_55], Z1[h > h0_55]))
    rho_h = (RHO_I_MGM * Z) / (1 + Z)
    rho = rho_h * 1000

    return rho
### end hl_analytic ###
#######################

def solver(a_U, a_D, a_P, b):
    '''
    Solve linear system Ax = b
    (for heat diffusion)
    '''
    nz = np.size(b)

    diags = (np.append([a_U, -a_P], [a_D], axis = 0))
    cols = np.array([1, 0, -1])

    big_A = spdiags(diags, cols, nz, nz, format = 'csc')
    big_A = big_A.T

    rhs = -b
    phi_t = splin.spsolve(big_A, rhs)

    return phi_t
### end solver ###
##################

def HeatDiffusion(z, dz, dti, Tz, rho):
    '''
    transient 1-d diffusion finite volume method
    Based on Patankar, 1980
    
    :param z: layer edges
    :param dz: layer thicknesses
    :param dti: time step size
    :param Tz: temperature profile
    :param rho: density profile

    :return Tz:
    '''

    midpoints = (z[:-1]+z[1:])/2 #mid points of layers
    
    Tz_s     = Tz[0] # temperature of upper layer, maintained as surface boundary condition
    c_vol    = rho * CP_I # specific heat of ice * firn density = heat capacity
    Gamma_P  = 0.021 + 2.5 * (rho/1000.0)**2  # thermal conductivity of firn

    ####################################### 
    deltaZ_u = np.diff(midpoints)
    deltaZ_u = np.append(deltaZ_u[0], deltaZ_u)
    deltaZ_d = np.diff(midpoints)
    deltaZ_d = np.append(deltaZ_d, deltaZ_d[-1])

    f_u = 1 - (midpoints[:] - z[0:-1]) / deltaZ_u[:]
    f_d = 1 - (z[1:] - midpoints[:]) / deltaZ_d[:]
       
    Gamma_U = np.append(Gamma_P[0], Gamma_P[0: -1] )
    Gamma_D = np.append(Gamma_P[1:], Gamma_P[-1])
    Gamma_u =  1 / ((1 - f_u) / Gamma_P + f_u / Gamma_U) # Patankar eq. 4.9
    Gamma_d =  1 / ((1 - f_d) / Gamma_P + f_d / Gamma_D)

    D_u = (Gamma_u / deltaZ_u)
    D_d = (Gamma_d / deltaZ_d)
    b_0 = 0 * dz # first term of Patankar eq. 4.41d

    a_U = D_u # Patankar eq. 4.41a,b
    a_D = D_d # Patankar eq. 4.41a,b
    a_P_0 = c_vol * dz / dti # (new) Patankar eq. 4.41c
    a_P     = a_U + a_D + a_P_0

    ### Boundary conditions:
    bc_u  = Tz_s # upper is the temperature of top layer
    bc_d  = 0     # lower is 0 gradient
    b = b_0 + a_P_0 * Tz #Patankar 4.41d
    #Upper boundary
    a_P[0]  = 1
    a_U[0]  = 0
    a_D[0]  = 0
    b[0]    = bc_u
    #Down boundary
    a_P[-1] = 1
    a_D[-1] = 0
    a_U[-1] = 1
    b[-1]   = deltaZ_u[-1] * bc_d      

    Tz = solver(a_U, a_D, a_P, b)

    return Tz
### end heat diffusion ###
##########################

def add_snowfall(snowfall_i, sf_mean, mass, dz, rho, rhos, Tz, Ts, dti):
    '''
    Adjust grid to accomodate new snowfall:
    puts new snow on top
    fluxes mass (equal to new snowfall mass) downward from one layer to the next
    temperature adjusted due to advection
    last layer fluxes out mass equal to long term accumulation rate
    layers have constant mass, except bottom layer mass fluctuates
    (flux into a layer is equal to flux out)

    :param snowfall_i: snowfall at this timestep [kg/m2/s]
    :param sf_mean: mean annual snowfall [kg/m2/s]
    :param mass: mass of each layer of snow/firn [kg]
    :param dz: thickness of each layer [m]
    :param rho: density of each layer [kg/m3]
    :param rhos: density of new snow accumulation [kg/m3]
    :param Tz: temperature profile
    :param Ts: temperature of new snowfall (Tskin at this time step)
    :param dti: time delta for this time step  
    '''
    
    mass_new = snowfall_i * dti # mass of new snow at this timestep [kg/m2]
    
    mass_out = sf_mean * dti # the mass removed is equal to the long term accumulation rate

    ### add new mass to top layer, and advect mass (equal to the new snow mass) downward through the layers
    rho_temp = np.append(rhos,rho[:-1]) # helper vector for calculations
    dz[:-1]  = dz[:-1] + (mass_new/rho_temp[:-1]) - mass_new/rho[:-1] # new thickness added to layer j is mass_new/density(j-1); mass removed is mass_new/density(j)
    dz[-1]   = dz[-1] + mass_new/rho_temp[-1] - (mass_out/rho[-1])# last node thickness removed is based on mass_out
    
    ### advect temperature by taking weighted mean of layer mass/temperature and mass/temperature of firn being fluxed in
    T_temp = np.append(Ts, Tz[:-1])  # helper vector for calculations
    Tz[:-1]  = ((mass_new*T_temp[:-1]) + (Tz[:-1]*(mass[:-1]-mass_new))) / mass[:-1] 
    Tz[-1]  = ((mass_new*T_temp[-1]) + (Tz[-1]*(mass[-1]-mass_out))) / (mass[-1] + mass_new - mass_out) # slightly different for bottom because mass out is different than mass in 
    
    mass[-1] = mass[-1] - mass_out + mass_new # mass of bottom laer
    
    ### adjust density and z based on dz, mass changes
    rho = mass/dz
    z = np.append(0,np.cumsum(dz))

    return dz, rho, z, mass_out/dti
### end add_snowfall ###
########################

def FDM(snowfall_mean,rho,Tz):
    '''
    Firn Densification Model

    presently the Herron and Langway model. Could import physics module and get different ones.
    Accumulation units in HL are m W.E. per year
    '''
    Q1  = 10160.0
    Q2  = 21400.0
    k1  = 11.0
    k2  = 575.0
    aHL = 1.0
    bHL = 0.5
    R = 8.314
    RHO_1 = 550.0
    RHO_I_MGM = 0.917

    A = snowfall_mean * SPY / 1000 # Accumulation in units m W.E. per year

    drho_dt = np.zeros_like(rho)
    drho_dt[rho < RHO_1]     = k1 * np.exp(-Q1 / (R * Tz[rho < RHO_1])) * (RHO_I_MGM - rho[rho < RHO_1] / 1000) * A**aHL * 1000 / SPY
    drho_dt[rho >= RHO_1]    = k2 * np.exp(-Q2 / (R * Tz[rho >= RHO_1])) * (RHO_I_MGM - rho[rho >= RHO_1] / 1000) * A**bHL * 1000 / SPY

    return drho_dt # densification rate, units kg/m3/s
### end FDM ###
###############


def generate_restarts(dz, forcings, repetitions=40, heat_diffusion=True, save_restart=True):
    '''
    Function to generate a restart file (initial condition) for a given climate forcing
    (could do some clever work to make forcing file name and restart file name have some characters that must match)
    '''
    nlayers = len(dz)
    rho = hl_analytic(dz,forcings['rhos'],forcings['T_mean'],forcings['snowfall_mean'])
    mass = rho * dz
    Tz = forcings['T_mean'] * np.ones_like(rho)
    try:
        irepeat       = np.where(forcings['time']<pd.to_datetime('1995,01,01'))[0] # reference climate
    except:
        irepeat = 365*15
    Tskin_spin    = np.tile(forcings['Tskin'][irepeat],repetitions)
    dt            = np.diff(forcings['time']).astype(float)/1e9 # seconds between time steps  
    snowfall_spin = np.tile(forcings['snowfall'][irepeat],repetitions)
    sf_mean       =  (snowfall_spin).mean()
    dt_spin       = np.tile(dt[irepeat],repetitions)
    for jj, dts in enumerate(dt_spin):
        ### First densify
        drho_dt = FDM(sf_mean,rho,Tz)
        rho = rho + drho_dt * dts
        dz = mass/rho
        z = np.append(0,np.cumsum(dz))
        ### then add new accumulation and adjust grid
        dz, rho, z, mass_out = add_snowfall(snowfall_spin[jj], sf_mean, mass, dz, rho, forcings['rhos'], Tz, Tskin_spin[jj],dts)
        if heat_diffusion:
            Tz = HeatDiffusion(z, dz, dts, Tz, rho)
    if save_restart:
        rs_time = forcings['time'][0]
        save_restarts(rs_time, nlayers, rho, dz, Tz)
   
    return dz,rho,Tz
### end generate_restarts ###
#############################

def save_restarts(rs_time, nlayers, rho, dz, Tz):
    '''
    Function to save the current state of the model as a restart for future use
    '''
    if type(rs_time)==np.datetime64:
        ts = pd.to_datetime(rs_time) 
        d = ts.strftime('%Y.%m.%d')
    else:
        d = rs_time
    layers = np.arange(0,nlayers)
    fn_out = f'restarts/miniCFM_restart_{d}.nc'
    ds = xr.Dataset(
        data_vars=dict(
            rho = (["layers","time"],rho[:,np.newaxis], {"units":'kg'}),
            dz  = (["layers","time"],dz[:,np.newaxis], {"units":'m'}),
            Tz  = (["layers","time"],Tz[:,np.newaxis], {"units":'K'}),
            ),
        coords = dict(
        layers = (["layers"],layers),
        time = np.atleast_1d(rs_time)
        )
            )
    ds.to_netcdf(fn_out)
    ds.close()
### end save_restarts ###
#########################

class MINICFM:
    '''
    class for the miniture, lightweight version of CFM
    '''

    def __init__(self,forcings,restarts = None):
        '''
        initialize: set up the grid, import forcings
        if rho is none, then no restarts have been provided 
        '''

        # load forcings; this is a dictionary.
        self.Tskin      = forcings['Tskin'] # skin temperature/assumed temperature of new precip [K]
        self.snowfall   = forcings['snowfall'] # snowfall rate [kg/m2/second]
        # self.t          = forcings['time'] # model dates, vector python datetime64, may change in future  
        self.T_mean     = forcings['T_mean'] # mean annual T at site [K]
        self.sf_mean    = forcings['snowfall_mean'] # mean annual snowfall [kg/m2/s]
        self.rhos       = forcings['rhos'] # new snow density [kg/m3]
        # self.dt         = np.diff(self.t).astype(float)/1e9 # seconds between time steps

        ### initial state from restarts
        self.dz      = restarts['dz']
        self.nlayers = len(self.dz)
        self.z       = np.append(0,np.cumsum(self.dz))
        self.rho     = restarts['rho']
        self.mass    = self.rho * self.dz
        try:
            self.Tz = restarts['Tz']
        except:
            self.Tz = self.T_mean * np.ones_like(self.rho)
    ### end __init__ ###
    ####################
   
    def miniCFM(self,iii,dti,heat_diffusion=True):
        ### First densify
        drho_dt = FDM(self.sf_mean,self.rho,self.Tz)
        self.rho = self.rho + drho_dt * dti
        self.dz = self.mass/self.rho
        self.z = np.append(0,np.cumsum(self.dz))
        ### then add new accumulation, flux out bottom, and adjust grid
        self.dz, self.rho, self.z, mass_out = add_snowfall(self.snowfall[iii], self.sf_mean, self.mass, self.dz, self.rho, self.rhos,self.Tz, self.Tskin[iii], dti)
        if heat_diffusion:
            self.Tz = HeatDiffusion(self.z, self.dz, dti, self.Tz, self.rho)
        
        return self.dz,self.rho,self.mass,self.Tz, mass_out
    ### end miniCFM ###
    ###################

if __name__ == '__main__':
    
    ################################################
    ### Get boundary conditions (forcings)

    ### example 1: import from a dataframe, with date time index
    # f_in = pd.read_pickle('inputs/MERRA2_CLIM_df_72.5_-36.25.pkl')
    # forcing = f_in.ffill()
    # snowfall_mean = (forcing.ffill()['PRECSNO']/(86400)).mean() # mean snowfall rate, kg/m2/s
    # T_mean = forcing['TS'].mean()
    # rhos = 315.0
    # ftime = forcing.index.values
    # fdt = np.diff(ftime).astype(float)/1e9
    # ### forcing['PRECTOT'] from my input has units of kg/m2/day. what goes to model needs to be kg/m2/second.
    # forcings = {'time':ftime,'dt':fdt,'Tskin':forcing['TS'].values,'snowfall':forcing['PRECTOT'].values/(24*3600),'T_mean':T_mean,'snowfall_mean':snowfall_mean,'rhos':rhos}
    
    ### example 2: generate synthetic climate, with arbitrary model time
    # use createForcing.py to make a forcings
    with xr.open_dataset('inputs/miniCFM_forcing_example.nc') as ds_in:
        forcings = {}
        forcings['time'] = ds_in['time'].values
        forcings['Tskin']    = ds_in['Tskin'].values
        forcings['snowfall'] = ds_in['snowfall'].values
        forcings['snowfall_mean'] = ds_in['snowfall_mean'].values
        forcings['T_mean'] = ds_in['T_mean'].values
        forcings['rhos'] = ds_in['rhos'].values
        forcings['dt'] = ds_in['dt'].values * np.ones(len(forcings['time'])-1)
    ################################################

    ### get initial conditions:
    ### either spin up a new run and create a restart, or initialize using an existing restart
    new_restarts = True # whether to use an existing restart or spin up a new model
    restart_fn = 'restarts/miniCFM_restart_1980.01.01.nc'
    ### (still coming up with a clever way to make forcing file and restart file have consistent name scheme)
    rse = os.path.exists(restart_fn)
    if ((not new_restarts) and (rse)): # use exsiting restart
        with xr.open_dataset(restart_fn) as rs:
            start_time = rs.time.values
            restarts = {}
            restarts['rho'] = rs['rho'].values.flatten()
            restarts['dz'] = rs['dz'].values.flatten()      
            istart = np.where(forcings['time']>=start_time)[0][0]
            if istart!=0:
                ip1 = istart
                forcings['time'] = forcings['time'][ip1:]
                forcings['Tskin'] = forcings['Tskin'][ip1:]
                forcings['snowfall'] = forcings['snowfall'][ip1:]
    
    else: # spin up a model run, optionally save the restart for future use
        dz = np.array([0.1,1,5,10,30,75]) # initial thicknesses, will change a bit
        init_dz, init_rho, init_Tz = generate_restarts(dz,forcings,repetitions=10,save_restart=True)
        restarts = {'dz':init_dz,'rho':init_rho,'Tz':init_Tz}
    ############

    ### Model run
    cmini = MINICFM(forcings = forcings,restarts=restarts) # initialize model run
    modeltimes = forcings['time'] # model dates, vector python datetime64, may change in future
    dt = forcings['dt'] # seconds between time steps (if you change modeltime format, this must stay as seconds)
    
    rdict = {} # results dictionary
    rdict['rho'] = np.ones((len(modeltimes)-1,len(cmini.dz)))
    rdict['dz'] = np.ones((len(modeltimes)-1,len(cmini.dz)))
    rdict['mass'] = np.ones((len(modeltimes)-1,len(cmini.dz)))
    rdict['Tz'] = np.ones((len(modeltimes)-1,len(cmini.dz)))
    rdict['mass_out'] = np.ones(len(modeltimes)-1)
    rdict['modeltime'] = []

    for iii,dti in enumerate(dt): # time stepping model loop
        modeltime = modeltimes[iii+1] # the time for this model step, +1 because we iterate through dt, and the model results at the end of the iteration are at the end of the time step.
        ts = pd.to_datetime(modeltime) 
        d = ts.strftime('%Y.%m.%d')

        dz, rho, mass, Tz, mass_out = cmini.miniCFM(iii,dti) # run the model for this time step
        
        ### example: save a restart at some timestep
        if iii==400: 
            save_restarts(modeltime, len(rho), rho, dz, Tz)

        ### example: change a value within the timestepping loop, e.g., DA could adjust density or dz
        # if modeltime==np.datetime64('2020-03-12'): 
        if modeltime == 801:
            cmini.rho = 800*np.ones_like(rho) 
            cmini.dz = cmini.mass / cmini.rho # need to make sure that layer mass/dz/density stay self consistent. Currently coded so that layer mass is constant.
        
        ### put time step outputs in results dictionary
        rdict['rho'][iii,:]  = cmini.rho
        rdict['mass'][iii,:] = cmini.mass
        rdict['dz'][iii,:]   = cmini.dz
        rdict['Tz'][iii,:]   = cmini.Tz
        rdict['mass_out'][iii] = mass_out
        rdict['modeltime'].append(d)
    ############
    
    ### save results
    layers    = np.arange(0,len(cmini.rho))
    modeltime = pd.DatetimeIndex(rdict['modeltime'])
    rds = xr.Dataset(
        data_vars = dict(
            rho  = (["time","layers"],rdict['rho'], {"units":'kg/m3'}),
            dz   = (["time","layers"],rdict['dz'], {"units":'m'}),
            mass = (["time","layers"],rdict['mass'], {"units":'kg'}),
            Tz   = (["time","layers"],rdict['Tz'], {"units":'K'}),
            flux_out = (["time"],rdict['mass_out'], {"units":'kg'}),
            ),
        coords = dict(
        layers = (["layers"],layers),
        time = modeltime
        )
            )
    fn_out = 'outputs/CFM_outputs.nc'
    rds.to_netcdf(fn_out)
    rds.close()
    ############

'''
GSFC FDM code. ignore for now.
        ### Model constants
        ar1 = 0.07
        ar2 = 0.03
        Eg  = 42.4e3
        Ec1  = 59500.0
        Ec2 = 56870.0
        alpha1 = 0.91
        alpha2 = 0.644
        R = 8.314
        RHO_1 = 550.0
        RHO_I = 917.0
        ###
        bdot_mean = self.snowfall[self.iii] / SPY # temporary
        A_mean_1 = bdot_mean
        A_mean_2 = bdot_mean
        c1 = ar1 * A_mean_1**alpha1 * GRAVITY * np.exp(-Ec1 / (R * self.Tz[self.rho < RHO_1]) + Eg / (R * self.T_mean[self.iii]))
        c2 = ar2 * A_mean_2**alpha2 * GRAVITY * np.exp(-Ec2 / (R * self.Tz[self.rho >= RHO_1]) + Eg / (R * self.T_mean[self.iii]))
        dr_dt = np.ones_like(self.dz)
        dr_dt[self.rho < RHO_1]  = c1 * (RHO_I - self.rho[self.rho < RHO_1])
        dr_dt[self.rho >= RHO_1] = c2 * (RHO_I - self.rho[self.rho >= RHO_1])
        drho_dt = dr_dt / SPY # densification rate, kg/m3 per second
        return drho_dt
'''
