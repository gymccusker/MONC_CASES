###
###
### SCRIPT TO READ IN ASCOS RADIOSONDE DATA AND OUTPUT FOR MONC
###
###

from __future__ import print_function
import time
import datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import os
import seaborn as sns
from scipy.interpolate import interp1d

#### import python functions
import sys
sys.path.insert(1, '../py_functions/')
from time_functions import calcTime_Mat2DOY, calcTime_Date2DOY
from readMAT import readMatlabStruct, readMatlabData
from physFuncts import calcThetaE, calcThetaVL, adiabatic_lwc, calcTemperature, calcAirDensity
from pyFixes import py3_FixNPLoad

def quicklooksSonde(data):

    '''
    Quicklooks from Jutta's radiosonde file
    '''

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(12,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.92, left = 0.1,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.5e3

    # index = 1

    plt.subplot(131)
    plt.plot(data['sonde']['temperature'][:] + 273.15, data['sonde']['Z'])
    plt.ylabel('Z [m]')
    plt.xlabel('Temperature [K]')
    plt.ylim([0,yylim])
    plt.grid('on')
    plt.xlim([259,275])

    plt.subplot(132)
    plt.plot(data['sonde']['sphum'][:], data['sonde']['Z'], label = 'sphum')
    plt.plot(data['sonde']['mr'][:], data['sonde']['Z'], label = 'mr')
    plt.xlabel('[g/kg]')
    plt.grid('on')
    plt.legend()
    plt.ylim([0,yylim])
    plt.title('MOCCHA SONDE DOY = ' + str(data['sonde']['doy'][:]))

    plt.subplot(133)
    plt.plot(data['sonde']['RH'][:], data['sonde']['Z'])
    plt.xlabel('Rel. Hum. [%]')
    plt.grid('on')
    plt.ylim([0,yylim])

    # plt.savefig('../MOCCHA/FIGS/Quicklooks_20180913-0000.png')
    plt.show()

    # print (data['sonde']['temperature'][0,index] + 273.16)

def sondeTHREF(data):

    '''
    Load initialisation reference potential temperature profile
    '''
    #### dz is 5m between 0 and 1157.5 m, then linearly interpolated to 10m between 1285m and 2395m
    data['pressure'] = data['sonde']['pressure']*1e2
    data['temperature'] = data['sonde']['temperature'] + 273.16
    data['q'] = data['sonde']['sphum']/1e3
    data['z'] = data['sonde']['Z']

    data['theta'], data['thetaE'] = calcThetaE(data['temperature'], data['pressure'], data['q'])

    ### print out surface values for monc namelist
    print (data['z'][0])
    print (data['theta'][0,1])
    print (data['pressure'][0,1])

    ####    --------------- FIGURE
    #
    # SMALL_SIZE = 12
    # MED_SIZE = 14
    # LARGE_SIZE = 16
    #
    # plt.rc('font',size=MED_SIZE)
    # plt.rc('axes',titlesize=MED_SIZE)
    # plt.rc('axes',labelsize=MED_SIZE)
    # plt.rc('xtick',labelsize=MED_SIZE)
    # plt.rc('ytick',labelsize=MED_SIZE)
    # plt.figure(figsize=(4,5))
    # plt.rc('legend',fontsize=MED_SIZE)
    # plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.25,
    #         hspace = 0.22, wspace = 0.4)
    #
    # yylim = 2.4e3
    #
    # plt.plot(data['ascos1']['thref'], data['ascos1']['z'], label = 'ASCOS1')
    # plt.plot(data['theta'], data['z'], label = 'SONDE')
    # plt.ylabel('Z [m]')
    # plt.xlabel('$\Theta$ [K]')
    # plt.ylim([0,yylim])
    # plt.xlim([265,295])
    #
    # plt.savefig('../FIGS/Quicklooks_LEM-ASCOS1_' + sondenumber + '.png')
    # plt.show()

    return data

def sondeTHINIT_QINIT1(data):


    '''
    Calculate initialisation potential temperature and moisture profiles
    '''

        ### build new height array for namelist initialisation, 50m up to 1k then 100m above
    ###         add damping layer between 2 and 2.5 km
    nml_Z = np.arange(0., 1000., 50.)
    nml_Z = np.append(nml_Z, np.arange(1000., 2501., 100.))
    print (nml_Z)

    ### build qinit1 array
    print (np.squeeze(data['sonde']['Z'][4:]).shape)
    print (data['sonde']['sphum'][4:].shape)

    interp_qinit1 = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde']['sphum'][:])
    nml_qinit1 = interp_qinit1(nml_Z[1:])

    ### build thref array
    interp_thref = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde']['pottemp'][:] + 273.16)
    nml_thref = interp_thref(nml_Z[1:])

    ### manually append last value to 2400m (since last Z in ASCOS1 is 2395m and above interpolation range)
    # nml_Z = np.append(nml_Z, 2400.)
    # nml_qinit1 = np.append(nml_qinit1, data['ascos1']['qinit1'][-1])
    # nml_thref = np.append(nml_thref, data['ascos1']['thref'][-1])

    ### save to dictionary so data can be easily passed to next function
    data['monc'] = {}
    data['monc']['z'] = nml_Z
    data['monc']['thref'] = nml_thref
    data['monc']['thinit'] = nml_thref
    data['monc']['qinit1'] = nml_qinit1


    ####    --------------- FIGURE

    # SMALL_SIZE = 12
    # MED_SIZE = 14
    # LARGE_SIZE = 16
    #
    # plt.rc('font',size=MED_SIZE)
    # plt.rc('axes',titlesize=MED_SIZE)
    # plt.rc('axes',labelsize=MED_SIZE)
    # plt.rc('xtick',labelsize=MED_SIZE)
    # plt.rc('ytick',labelsize=MED_SIZE)
    # plt.figure(figsize=(8,5))
    # plt.rc('legend',fontsize=MED_SIZE)
    # plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.1,
    #         hspace = 0.22, wspace = 0.4)
    #
    # yylim = 2.5e3
    #
    # plt.subplot(121)
    # # plt.plot(data['ascos1']['thinit'], data['ascos1']['z'], label = 'ASCOS1')
    # plt.plot(data['sonde']['pottemp'][:] + 273.16, data['sonde']['Z'], label = 'SONDE')
    # plt.plot(nml_thref, nml_Z[1:], 'k.', label = 'monc-namelist')
    # plt.ylabel('Z [m]')
    # plt.xlabel('$\Theta$ [K]')
    # plt.ylim([0,yylim])
    # # plt.xlim([265,295])
    #
    # plt.subplot(122)
    # # plt.plot(data['ascos1']['qinit1'][data['ascos1']['qinit1'] > 0], data['ascos1']['z'][data['ascos1']['qinit1'] > 0], label = 'ASCOS1')
    # plt.plot(data['sonde']['sphum'][:], data['sonde']['Z'], label = 'SONDE')
    # plt.plot(nml_qinit1, nml_Z[1:], 'k.', label = 'monc-namelist')
    # plt.xlabel('q [kg/kg]')
    # plt.grid('on')
    # plt.ylim([0,yylim])
    # plt.legend()
    #
    # # plt.savefig('../MOCCHA/FIGS/Quicklooks_LEM-ASCOS1-MONCnmlist_thinit-qinit1_20180913.png')
    # plt.show()

    return data

def sondeWINDS(data):


    '''
    Calculate initialisation wind (u/v) profiles
    '''

    # print (data['sonde']['v'][:,1])
    # print (data['sonde']['u'][:,1])

    ### build v array
    nml_v = np.zeros(np.size(data['monc']['z']))
    interp_v = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde']['v'][:])
    nml_v[1:] = interp_v(data['monc']['z'][1:])
    nml_v[0] = data['sonde']['v'][0]

    ### build u array
    nml_u = np.zeros(np.size(data['monc']['z']))
    interp_u = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde']['u'][:])
    nml_u[1:] = interp_u(data['monc']['z'][1:])
    nml_u[0] = data['sonde']['u'][0]

    ### save to dictionary so data can be easily passed to next function
    data['monc']['u'] = nml_u
    data['monc']['v'] = nml_v

    ### mean winds for geostrophic input
    print(np.nanmean(data['monc']['u']))
    print(np.nanmean(data['monc']['v']))

    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(8,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.1,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.5e3

    plt.subplot(121)
    # plt.plot(data['ascos1']['thinit'], data['ascos1']['z'], label = 'ASCOS1')
    plt.plot(data['sonde']['u'][:], data['sonde']['Z'], label = 'SONDE')
    plt.plot(data['monc']['u'], data['monc']['z'][:], 'k.', label = 'monc-namelist')
    plt.ylabel('Z [m]')
    plt.xlabel('u [m/s]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.xlim([-20,10])

    plt.subplot(122)
    # plt.plot(data['ascos1']['qinit1'][data['ascos1']['qinit1'] > 0], data['ascos1']['z'][data['ascos1']['qinit1'] > 0], label = 'ASCOS1')
    plt.plot(data['sonde']['v'][:], data['sonde']['Z'], label = 'SONDE')
    plt.plot(data['monc']['v'], data['monc']['z'][:], 'k.', label = 'monc-namelist')
    plt.xlabel('v [m/s]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.legend()
    plt.xlim([-20,20])

    # plt.savefig('../MOCCHA/FIGS/Quicklooks_winds_20180912T1800Z.png')
    plt.show()

    return data

def sondeQINIT2(data):

    '''
    Calculate adiabatic lwc up to 650m (main inversion):
        -- take thref and pressure, & calculate temperature
        # lwc_adiabatic(ii,liquid_bases(jj):liquid_tops(jj)) = dlwc_dz.*[1:(liquid_tops(jj)-liquid_bases(jj)+1)].*dheight;
    '''

    # print (np.squeeze(data['z']).shape)
    # print (data['pressure'][:,1].shape)
    print (data['monc']['z'][1:])

    data['monc']['pressure'] = np.zeros(np.size(data['monc']['z']))
    interp_pres = interp1d(np.squeeze(data['sonde']['Z']),np.squeeze(data['sonde']['pressure'][:])*1e2)
    data['monc']['pressure'][1:] = interp_pres(data['monc']['z'][1:])
    data['monc']['pressure'][0] = 100910. ## reference surface pressure from mcf

    # plt.plot(data['monc']['pressure'],data['monc']['z'])
    # plt.show()

    ### calculate temperature from thref and pressure
    temp_T = calcTemperature(data['monc']['thref'], data['monc']['pressure'][1:])

    ### adapt temperature array
    data['monc']['temperature'] = temp_T # np.zeros(np.size(data['monc']['z']))

    ### calculate qinit2
    ### interpolate free troposphere temperatures from radiosonde onto monc namelist gridding
    data['monc']['temperature'] = np.zeros(np.size(data['monc']['z']))
    interp_temp = interp1d(np.squeeze(data['sonde']['Z']), np.squeeze(data['sonde']['temperature'][:]) + 273.16)
    data['monc']['temperature'][1:] = interp_temp(data['monc']['z'][1:])
    data['monc']['temperature'][0] = data['sonde']['temperature'][0] + 273.15

    ### calculate adiabatic lwc rate of change
    dlwcdz, dqldz, dqdp = adiabatic_lwc(data['monc']['temperature'], data['monc']['pressure'])
    dheight = data['monc']['z'][1:] - data['monc']['z'][:-1]

    print (dheight.shape)
    print (dlwcdz.shape)

    ## define cloud layer
    freetrop_index = np.where(data['monc']['z'] >= 1000.0)
    dheight[int(freetrop_index[0][0]):] = 0.0   ## ignore points in the free troposphere
    blcloud_index = np.where(data['monc']['z'] < 200.0)
    dheight[blcloud_index] = 0.0   ## ignore points in the free troposphere

    ## calculate adiabatic cloud liquid water content
    data['monc']['qinit2'] = dlwcdz[:-1] * dheight
    data['monc']['qinit2'] = np.append(data['monc']['qinit2'], 0.)

    print (data['monc']['temperature'].shape)
    print (data['monc']['pressure'].shape)
    print (data['monc']['qinit1'].shape)
    print (data['monc']['qinit2'].shape)

    ### adapt theta (thinit and thref) based on revised temperature profile
    data['monc']['thref'] = np.zeros(np.size(data['monc']['z']))
    data['monc']['thinit'], thetaE = calcThetaE(data['monc']['temperature'][1:], data['monc']['pressure'][1:], data['monc']['qinit1'])
    data['monc']['thref'][1:] = data['monc']['thinit']
    data['monc']['thref'][0] = 267.27

    tempvar = data['monc']['qinit1']
    data['monc']['qinit1'] = np.zeros(np.size(data['monc']['z']))
    data['monc']['qinit1'][1:] = tempvar / 1e3
    data['monc']['qinit1'][0] = data['sonde']['sphum'][0] / 1e3

    ### combine q01 and q02 into one input field
    data['monc']['qinit'] = np.append(data['monc']['qinit1'], data['monc']['qinit2'])

    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(13,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.1,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.4e3

    plt.subplot(151)
    # ax = plt.gca()
    # ax.fill_between(data)
    plt.plot(data['sonde']['pottemp'][:] + 273.16, data['sonde']['Z'], color = 'darkorange', label = 'SONDE')
    plt.plot(data['monc']['thref'], data['monc']['z'][:], 'k.', label = 'monc-namelist')
    plt.ylabel('Z [m]')
    plt.xlabel('$\Theta$ [K]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.xlim([265,295])

    plt.subplot(152)
    plt.plot(data['sonde']['sphum'][:]/1e3, data['sonde']['Z'], color = 'darkorange', label = 'SONDE')
    plt.plot(data['monc']['qinit1'], data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('qinit1 [g/kg]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.xlim([0.2/1e3, 5./1e3])
    plt.legend(bbox_to_anchor=(0.25, 1.01, 1., .102), loc=3, ncol=3)

    plt.subplot(153)
    plt.plot(data['sonde']['pressure'][:]*1e2, data['sonde']['Z'], color = 'darkorange', label = 'SONDE')
    plt.plot(data['monc']['pressure'], data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('pressure [Pa]')
    plt.grid('on')
    plt.ylim([0,yylim])
    # plt.xlim([7e4, 10.5e4])

    plt.subplot(154)
    plt.plot(data['sonde']['temperature'][:] + 273.16, data['sonde']['Z'], color = 'darkorange', label = 'SONDE')
    plt.plot(data['monc']['temperature'], data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('temperature [K]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.xlim([265,275])

    plt.subplot(155)
    plt.plot(data['monc']['qinit2']*1e3, data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('qinit2 [g/kg]')
    plt.grid('on')
    plt.ylim([0,yylim])
    # plt.xlim([265,275])

    # plt.savefig('../MOCCHA/FIGS/Quicklooks_thref-qinit1-pres-temp-qinit2_MONCnmlist_20180912T1800Z.png')
    plt.show()

    return data

def aerosolACCUM(data):

    '''
    Design accummulation mode aerosol inputs:
        names_init_pl_q=accum_sol_mass, accum_sol_number
    '''

    print ('Designing soluble accummulation mode input:')

    arrlen = np.size(data['monc']['z'])
    print(arrlen)

    case = 'CASIM-UKCA-AeroProf' # 'CASIM-100'
    if case == 'CASIM-100':

        ### For UM_CASIM-100, the following were set:
        ###         accum_sol_mass_var=70*1.50e-9
        ###         accum_sol_num_var=70*1.00e8

        data['monc']['q_accum_sol_mass'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_mass'][:] = 1.50e-9
        print (data['monc']['q_accum_sol_mass'])

        data['monc']['q_accum_sol_number'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_number'][:] = 1.00e8
        print (data['monc']['q_accum_sol_number'])

    elif case == 'CASIM-UKCA-AeroProf':

        ### For UM_CASIM-100, the following were set:
        ###         accum_sol_mass_var=70*1.50e-9
        ###         accum_sol_num_var=70*1.00e8

        data['monc']['ukca'] = {}
        data['monc']['ukca'] = np.load('../MOCCHA/input/MONC_UKCAInputs-20180913.npy').item()


        data['monc']['q_accum_sol_number'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_number'][:] = data['monc']['ukca']['moncNumAccum']
        print (data['monc']['q_accum_sol_number'])

        rho_air = calcAirDensity(data['monc']['temperature'],data['monc']['pressure']/1e2)
        plt.plot(rho_air,data['monc']['z']);plt.show()

        modeFlag = 1 ### accumulation mode
        data['monc']['q_accum_sol_mass'] = estimateMass(data['monc']['q_accum_sol_number'][:], rho_air, modeFlag)
        print (data['monc']['q_accum_sol_mass'])

    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(10,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.1,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.4e3

    plt.subplot(121)
    plt.plot(data['monc']['q_accum_sol_number'], data['monc']['z'],'k.', label = 'MONC input')
    plt.ylabel('Z [m]')
    plt.xlabel('N$_{sol, accum}$ [m$^{-3}$]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.title('CASE = ' + case)
    # plt.xlim([0, 1.1e8])
    plt.legend()

    plt.subplot(122)
    plt.plot(data['monc']['q_accum_sol_mass'], data['monc']['z'],'k.')
    plt.ylabel('Z [m]')
    plt.xlabel('M$_{sol, accum}$ [kg/kg]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.title('CASE = ' + case)
    # plt.xlim([0, 1.1e8])

    plt.savefig('../MOCCHA/FIGS/20180913_NsolAccum_MsolAccum_' + case + '.png')
    plt.show()


    ### combine to existing q field input
    data['monc']['q_accum_sol'] = np.append(data['monc']['q_accum_sol_mass'], data['monc']['q_accum_sol_number'])
    data['monc']['qinit'] = np.append(data['monc']['qinit'], data['monc']['q_accum_sol'])


    return data

def estimateMass(N, rho_air, flag):

    #### -------------------------------------------------------------
    #### SCALE AEROSOL MASS (accumulation mode: 1.5*1e-9 for every 1.00*1e8 aerosol particles)
    #### -------------------------------------------------------------

    print('')
    print('****')
    print('Estimate mass by mean modal radius and assuming spherical particles:')
    print('')

    #### Accumulation mode: 0.1um < d_p < 1um

    ### make dummy variables
    # M = 1.0
    if flag == 1:
        sigma = 1.5         #### == fixed_aerosol_sigma (mphys_constants.F90)
        rho = 1777.0        #### == fixed_aerosol_density (mphys_constants.F90); kg/m3
        Rm = 0.5*1.0e-6     #### == fixed_aerosol_rm (mphys_constants.F90); 500nm
    elif flag == 2:
        sigma = 1.5         #### == fixed_aerosol_sigma (mphys_constants.F90)
        rho = 2000.0        #### == fixed_aerosol_density (mphys_constants.F90); kg/m3
        Rm = 5*1.0e-6       #### == fixed_aerosol_rm (mphys_constants.F90); 5 um
    else:
        print('****Mode option not valid!****')

    print('Calculating aerosol mass mixing ratio assuming: ')
    print('rho_aero = ', rho, ' kg/m3')
    print('Rm = ', Rm*1e6, ' um')
    print('...')

    ### calculation for mean radius given mass and number:
    # MNtoRm = ( 3.0*M*np.exp(-4.5*np.log(sigma)**2) /
    #     (4.0*N*np.pi*rho) )**(1.0/3.0)
                ### just copied from casim/lognormal_funcs.F90

    mass = ( (4.0/3.0)*np.pi*Rm**3 ) * (N*rho) / (np.exp(-4.5*np.log(sigma)**2))
            ### gives mass concentration in kg/m3

    #### need mass concentration in kg/kg for casim input
    M = mass / rho_air

    print('mass = ', M)
    print('')

    return M

def moncInput(data):


        print ('***')
        print ('***')
        ### print out to terminal in format for monc namelists
        print ('z_init_pl_theta = ')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_theta = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_theta = ')
        for line in data['monc']['thref']: sys.stdout.write('' + str(np.round(line,2)).strip() + ',')
        print ('')

                    # f_init_pl_theta = 267.27,267.58,267.59,267.57,267.65,267.74,268.06,268.91,269.5,269.95,270.35,270.53,270.72,271.07,271.46,272.2,273.29,273.68,274.34,274.53,274.62,275.2,276.39,276.7,276.77,276.89,279.95,282.59,282.9,283.5,284.72,286.12,287.18,287.81,288.38,288.95

        print ('z_init_pl_u = ')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_u = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_u = ')
        for line in data['monc']['u']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
        print ('')

                    # f_init_pl_u = -9.4125,-10.43,-11.875,-12.43,-12.55,-12.17,-11.38167,-10.9,-11.57,-12.7,-13.54,-13.45,-12.86,-11.93,-11.08333,-10.9,-11.28333,-11.95,-12.55,-12.7,-12.45,-10.83,-9.75,-9.6,-10.52333,-11.2,-11.2,-11.85,-11.7875,-11.28333,-10.96667,-11.35,-12.55,-12.03333,-11.55,-10.95

        print ('z_init_pl_v = ')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_v = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_v = ')
        for line in data['monc']['v']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
        print ('')

                    # f_init_pl_v = 0.825,1.74,2.85,2.9,3.26,3.93,4.4,4.11,3.03,2.5,3.2775,4.16,4.72,4.75,4.05,2.46667,1.08333,0.6,1.15,2.13167,2.85,2.93,0.11,0.01333,0.8,-0.40333,-3.175,-2.7,-3.0,-2.9,-2.95,-3.45,-3.0,-4.15,-4.45,-4.6

        print ('z_init_pl_q = ')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_q = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_q = ')
        for line in data['monc']['qinit']: sys.stdout.write('' + str(np.round(line,10)).strip() + ',')
        print ('')

                    # f_init_pl_q = 0.00244,0.00227,0.00228,0.0023,0.00221,0.00214,0.00213,0.00215,0.00215,0.00212,0.00213,0.00211,0.00207,0.00203,0.00202,0.00199,0.00201,0.00201,0.00201,0.00204,0.00202,0.00198,0.00206,0.00206,0.00202,0.00201,0.00243,0.00197,0.00184,0.00184,0.00142,0.00107,0.00068,0.00062,0.00054,0.00042,0.0,0.0,0.0,0.0,6e-05,6e-05,6e-05,6e-05,6e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,



        print ('***')
        print ('***')

        return data

def main():

    START_TIME = time.time()
    print ('******')
    print ('')
    print ('Start: ' + time.strftime("%c"))
    print ('')

    '''
    Python script to build initialisation data for MONC from radiosondes
    '''

    print ('Import MOCCHA radiosonde data:')
    print ('...')

    print ('Load radiosonde data from Jutta...')
    obs_root_dir = '/home/gillian/MOCCHA/ODEN/DATA/'
    sondes = readMatlabStruct(obs_root_dir + 'radiosondes/SondeData_h10int_V02.mat')

    print ('')
    print (sondes.keys())

    ## -------------------------------------------------------------
    ## Choose sonde for initialisation:
    ## -------------------------------------------------------------
    sonde_option = '20180913T0000'#'20180912T1800' #

    if sonde_option == '20180912T1800':
        ## -------------------------------------------------------------
        ## Load radiosonde from 20180912 1800UTC
        ## -------------------------------------------------------------
        index256 = np.where(np.round(sondes['doy'][:,:]) == 256.)
        print (sondes['doy'][:,index256[1][0]])
        data = {}
        data['sonde'] = {}
        for k in sondes.keys():
            if k == 'Z': continue
            data['sonde'][k] = sondes[k][:,index256[1][0]]
        data['sonde']['Z'] = sondes['Z']

    elif sonde_option == '20180913T0000':
        ## -------------------------------------------------------------
        ## Load radiosonde from 20180913 0000UTC
        ## -------------------------------------------------------------
        index256 = np.where(np.round(sondes['doy'][:,:]) == 256.)
        print (sondes['doy'][:,index256[1][1]])
        data = {}
        data['sonde'] = {}
        for k in sondes.keys():
            if k == 'Z': continue
            data['sonde'][k] = sondes[k][:,index256[1][1]]
        data['sonde']['Z'] = sondes['Z']

    print (data['sonde'].keys())
    print (data['sonde']['doy'][:])

    ## -------------------------------------------------------------
    ## Quicklook plots of chosen sonde
    ## -------------------------------------------------------------
    # figure = quicklooksSonde(data)

    ## -------------------------------------------------------------
    ## Read in data from LEM namelists
    ##-------------------------------------------------------------
    # data = sondeTHREF(data)
    data = sondeTHINIT_QINIT1(data)
    data = sondeWINDS(data)

    ### design qfield inputs
    data = sondeQINIT2(data)
    data = aerosolACCUM(data)

    ## -------------------------------------------------------------
    ## Print out data in monc namelist format
    ## -------------------------------------------------------------
    data = moncInput(data)

    ## -------------------------------------------------------------
    ## save out working data for testing
    ## -------------------------------------------------------------
    np.save('working_data', sondes)

    # -------------------------------------------------------------
    # FIN.
    # -------------------------------------------------------------
    END_TIME = time.time()
    print ('******')
    print ('')
    print ('End: ' + time.strftime("%c"))
    print ('')


if __name__ == '__main__':

    main()
