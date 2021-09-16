###
###
### SCRIPT TO READ IN ASCOS RADIOSONDE DATA AND OUTPUT FOR MONC
###
###     test change

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
from time_functions import calcTime_Mat2DOY, calcTime_Date2DOY, serial_date_to_doy
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

    ### build qinit1 array (in kg/kg)
    print (np.squeeze(data['sonde']['Z'][4:]).shape)
    print (data['sonde']['sphum'][4:].shape)

    nml_qinit1 = np.zeros(np.size(nml_Z))
    interp_qinit1 = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde']['mr'][:]/1e3)
    nml_qinit1[1:] = interp_qinit1(nml_Z[1:])
    nml_qinit1[0] = data['sonde']['mr'][0]/1e3

    ### build thref array (in K)
    nml_thinit = np.zeros(np.size(nml_Z))
    interp_thinit = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde']['pottemp'][:] + 273.16)
    nml_thinit[1:] = interp_thinit(nml_Z[1:])
    nml_thinit[0] = data['sonde']['pottemp'][0] + 273.16

    ### manually append last value to 2400m (since last Z in ASCOS1 is 2395m and above interpolation range)
    # nml_Z = np.append(nml_Z, 2400.)
    # nml_qinit1 = np.append(nml_qinit1, data['ascos1']['qinit1'][-1])
    # nml_thref = np.append(nml_thref, data['ascos1']['thref'][-1])

    ### save to dictionary so data can be easily passed to next function
    data['monc'] = {}
    data['monc']['z'] = nml_Z
    # data['monc']['thref'] = nml_thref
    data['monc']['thinit'] = nml_thinit
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
    # plt.figure(figsize=(8,5))
    # plt.rc('legend',fontsize=MED_SIZE)
    # plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.1,
    #         hspace = 0.22, wspace = 0.4)
    #
    # yylim = 2.5e3
    #
    # plt.subplot(121)
    # # plt.plot(data['ascos1']['thinit'], data['ascos1']['z'], label = 'ASCOS1')
    # plt.plot(data['sonde']['u'][:], data['sonde']['Z'], label = 'SONDE')
    # plt.plot(data['monc']['u'], data['monc']['z'][:], 'k.', label = 'monc-namelist')
    # plt.ylabel('Z [m]')
    # plt.xlabel('u [m/s]')
    # plt.grid('on')
    # plt.ylim([0,yylim])
    # plt.xlim([-20,10])
    #
    # plt.subplot(122)
    # # plt.plot(data['ascos1']['qinit1'][data['ascos1']['qinit1'] > 0], data['ascos1']['z'][data['ascos1']['qinit1'] > 0], label = 'ASCOS1')
    # plt.plot(data['sonde']['v'][:], data['sonde']['Z'], label = 'SONDE')
    # plt.plot(data['monc']['v'], data['monc']['z'][:], 'k.', label = 'monc-namelist')
    # plt.xlabel('v [m/s]')
    # plt.grid('on')
    # plt.ylim([0,yylim])
    # plt.legend()
    # plt.xlim([-20,20])
    #
    # # plt.savefig('../MOCCHA/FIGS/Quicklooks_winds_20180912T1800Z.png')
    # plt.show()

    return data

def sondeQINIT2(data):

    '''
    Calculate adiabatic lwc up to 1km (main inversion):
        -- take thref and pressure, & calculate temperature
        # lwc_adiabatic(ii,liquid_bases(jj):liquid_tops(jj)) = dlwc_dz.*[1:(liquid_tops(jj)-liquid_bases(jj)+1)].*dheight;
    '''

    # print (np.squeeze(data['z']).shape)
    # print (data['pressure'][:,1].shape)
    print (data['monc']['z'][1:])

    ### define pressure array
    data['monc']['pressure'] = np.zeros(np.size(data['monc']['z']))
    interp_pres = interp1d(np.squeeze(data['sonde']['Z']),np.squeeze(data['sonde']['pressure'][:])*1e2)
    data['monc']['pressure'][1:] = interp_pres(data['monc']['z'][1:])
    data['monc']['pressure'][0] = 100910. ## reference surface pressure from mcf

    # plt.plot(data['monc']['pressure'],data['monc']['z'])
    # plt.show()

    ### calculate temperature from thref and pressure
    # data['monc']['temperature'] = calcTemperature(data['monc']['thref'], data['monc']['pressure'])

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
    freetrop_index = np.where(data['monc']['z'] >= 800.0)
    dheight[int(freetrop_index[0][0]):] = 0.0   ## ignore points in the free troposphere
    blcloud_index = np.where(data['monc']['z'] < 200.0)
    dheight[blcloud_index] = 0.0   ## ignore points towards the surface

    ## calculate adiabatic cloud liquid water mixing ratio
    data['monc']['qinit2'] = dqldz[:-1] * dheight
    data['monc']['qinit2'] = np.append(data['monc']['qinit2'], 0.)

    print (data['monc']['temperature'].shape)
    print (data['monc']['pressure'].shape)
    print (data['monc']['qinit1'].shape)
    print (data['monc']['qinit2'].shape)

    ### adapt theta (thinit and thref) based on revised temperature profile
    data['monc']['thref'] = np.zeros(np.size(data['monc']['z']))
    data['monc']['thinit'], thetaE = calcThetaE(data['monc']['temperature'], data['monc']['pressure'], data['monc']['qinit1'])
    data['monc']['thref'] = data['monc']['thinit']
    # data['monc']['thref'][0] = 267.27

    ### change qinit1 input to kg/kg
    # tempvar = data['monc']['qinit1']
    # data['monc']['qinit1'] = np.zeros(np.size(data['monc']['z']))
    # data['monc']['qinit1'] = tempvar / 1e3
    # data['monc']['qinit1'][0] = data['sonde']['sphum'][0] / 1e3

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
    plt.plot(data['sonde']['sphum'][:], data['sonde']['Z'], color = 'darkorange', label = 'SONDE')
    plt.plot(data['monc']['qinit1']*1e3, data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('qinit1 [g/kg]')
    plt.grid('on')
    plt.ylim([0,yylim])
    # plt.xlim([0.2/1e3, 5./1e3])
    plt.legend(bbox_to_anchor=(0.25, 1.01, 1., .102), loc=3, ncol=3)

    plt.subplot(153)
    plt.plot(data['sonde']['pressure'][:]*1e2, data['sonde']['Z'], color = 'darkorange', label = 'SONDE')
    plt.plot(data['monc']['pressure'], data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('pressure [Pa]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.xlim([7e4, 10.5e4])

    plt.subplot(154)
    plt.plot(data['sonde']['temperature'][:] + 273.16, data['sonde']['Z'], color = 'darkorange', label = 'SONDE')
    plt.plot(data['monc']['temperature'], data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('temperature [K]')
    plt.grid('on')
    plt.ylim([0,yylim])
    plt.xlim([260,275])

    plt.subplot(155)
    plt.plot(data['monc']['qinit2']*1e3, data['monc']['z'], 'k.', label = 'monc-namelist')
    plt.xlabel('qinit2 [g/kg]')
    plt.grid('on')
    plt.ylim([0,yylim])
    # plt.xlim([265,275])

    plt.savefig('../MOCCHA/FIGS/Quicklooks_thref-qinit1-pres-temp-qinit2-1km_MONCnmlist_' + data['sonde_option'] + '.png')
    plt.show()

    return data

def aerosolACCUM(data):

    '''
    Design accummulation mode aerosol inputs:
        names_init_pl_q=accum_sol_mass, accum_sol_number
    '''

    print ('Designing soluble accummulation mode input:')

    data['qAccum_flag'] = 1

    arrlen = np.size(data['monc']['z'])
    print(arrlen)

    case = 'CASIM-20'
        ### 'CASIM-0' - initialising qfields only
        ### 'CASIM-20' - 20/cc at all Z
        ### 'CASIM-100' - as Young et al., 2021
        ### 'CASIM-UKCA-AeroProf' - as Young et al., 2021
        ### 'CASIM-UKCA' - using Alberto's UKCA inputs

    if case == 'CASIM-0':

        ### For UM_CASIM-100, the following were set:
        ###         accum_sol_mass_var=70*1.50e-9
        ###         accum_sol_num_var=70*1.00e8

        data['monc']['q_accum_sol_mass'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_mass'][:] = 0.0
        print (data['monc']['q_accum_sol_mass'])

        data['monc']['q_accum_sol_number'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_number'][:] = 0.0
        print (data['monc']['q_accum_sol_number'])

    elif case == 'CASIM-20':

        data['monc']['q_accum_sol_mass'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_mass'][:] = 3.0e-10
        print (data['monc']['q_accum_sol_mass'])

        data['monc']['q_accum_sol_number'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_number'][:] = 2.00e7
        print (data['monc']['q_accum_sol_number'])

    elif case == 'CASIM-100':

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

        ### Load aerosol data from Ruth

        data['monc']['ukca-aeroprof'] = {}
        data['monc']['ukca-aeroprof'] = np.load('../MOCCHA/input/MONC_UKCAInputs-20180913.npy').item()

        data['monc']['q_accum_sol_number'] = np.zeros(arrlen)
        data['monc']['q_accum_sol_number'][:] = data['monc']['ukca-aeroprof']['moncNumAccum']
        print (data['monc']['q_accum_sol_number'])

        rho_air = calcAirDensity(data['monc']['temperature'],data['monc']['pressure']/1e2)
        plt.plot(rho_air,data['monc']['z']);plt.show()

        modeFlag = 1 ### accumulation mode
        data['monc']['q_accum_sol_mass'] = estimateMass(data['monc']['q_accum_sol_number'][:], rho_air, modeFlag)
        print (data['monc']['q_accum_sol_mass'])

    elif case == 'CASIM-UKCA':

        ### Load aerosol data from Alberto

        data['ukca'] = Dataset('../../../UKCA/DATA/2018_aug_sep_aerosol__cg495.nc','r')

        ### Fields are as follows:
        ###         field2207 = ACCUMULATION MODE (SOLUBLE) NUMBER
        ###         field2208 = ACCUMULATION MODE (SOL) H2SO4 MMR
        ###         field2209 = ACCUMULATION MODE (SOL) BC MMR
        ###         field2210 = ACCUMULATION MODE (SOL) OM MMR
        ###         field2211 = ACCUMULATION MODE (SOL) SEA SALT MMR
        ###         field2213 = COARSE MODE (SOLUBLE) NUMBER
        ###         field2214 = COARSE MODE (SOLUBLE) H2SO4 MMR
        ###         field2215 = COARSE MODE (SOLUBLE) BC MMR
        ###         field2216 = COARSE MODE (SOLUBLE) OM MMR
        ###         field2217 = COARSE MODE (SOLUBLE) SEA SALT MMR
        ###         field1634 = Dust division 1 mass mixing ratio
        ###         field1634_1 = Dust division 2 mass mixing ratio
        ###         field1634_2 = Dust division 3 mass mixing ratio
        ###         field1634_3 = Dust division 4 mass mixing ratio
        ###         field1634_4 = Dust division 5 mass mixing ratio
        ###         field1634_5 = Dust division 6 mass mixing ratio

        data['monc']['ukca'] = {}
        data['monc']['ukca']['naer_sol_accum'] = data['ukca']['field2207'][:]
        data['monc']['ukca']['maer_sol_accum'] = data['ukca']['field2208'][:] + data['ukca']['field2209'][:] + data['ukca']['field2210'][:] + data['ukca']['field2211'][:]

        data['monc']['ukca']['naer_sol_coarse'] = data['ukca']['field2213'][:]

        srl_nos = data['ukca'].variables['t'][:].data
        data['monc']['ukca']['doy'] = np.zeros(len(data['ukca']['t'][:].data))
        for i in range(0,len(srl_nos)): data['monc']['ukca']['doy'][i] = serial_date_to_doy(np.float(srl_nos[i]))

        # plt.figure()
        # plt.plot(np.nanmean(np.nanmean(np.squeeze(data['monc']['ukca']['naer_sol_accum'][0,:,:,-2:]),2),1),
        #     data['ukca'].variables['hybrid_ht'])
        # plt.ylim([0,1e4])
        # plt.show()

        ####        FIGURE

        SMALL_SIZE = 12
        MED_SIZE = 14
        LARGE_SIZE = 16

        plt.rc('font',size=MED_SIZE)
        plt.rc('axes',titlesize=LARGE_SIZE)
        plt.rc('axes',labelsize=LARGE_SIZE)
        plt.rc('xtick',labelsize=LARGE_SIZE)
        plt.rc('ytick',labelsize=LARGE_SIZE)
        plt.rc('legend',fontsize=LARGE_SIZE)
        fig = plt.figure(figsize=(8,6))
        plt.subplots_adjust(top = 0.93, bottom = 0.1, right = 0.82, left = 0.08,
                hspace = 0.3, wspace = 0.1)

        plt.subplot(211)
        ax = plt.gca()
        img = plt.pcolormesh(data['monc']['ukca']['doy'],data['ukca'].variables['hybrid_ht'][:],
            np.transpose(np.nanmean(np.nanmean(data['monc']['ukca']['naer_sol_accum'][:,:,-2:,:],3),2)),
            # vmin = 0, vmax = 0.3
            )
        plt.ylim([0, 1e4])
        ax.set_xlim([226,258])
        # plt.xticks([230,235,240,245,250,255])
        # ax.set_xticklabels(['18 Aug','23 Aug','28 Aug','2 Sep','7 Sep','12 Sep'])
        plt.ylabel('Z [km]')
        plt.ylim([0,9000])
        axmajor = np.arange(0,9.01e3,3.0e3)
        axminor = np.arange(0,9.01e3,0.5e3)
        plt.yticks(axmajor)
        ax.set_yticklabels([0,3,6,9])
        ax.set_yticks(axminor, minor = True)
        cbaxes = fig.add_axes([0.85, 0.6, 0.015, 0.3])
        cb = plt.colorbar(img, cax = cbaxes, orientation = 'vertical')
        plt.ylabel('N$_{aer, sol, coarse}$ [cm$^{-3}$]', rotation = 270, labelpad = 25)

        plt.subplot(212)
        ax = plt.gca()
        img = plt.pcolormesh(data['monc']['ukca']['doy'],data['ukca'].variables['hybrid_ht'][:],
            np.transpose(np.nanmean(np.nanmean(data['monc']['ukca']['naer_sol_coarse'][:,:,-2:,:],3),2)),
            # vmin = 0, vmax = 200
            )
        plt.ylim([0, 1e4])
        ax.set_xlim([226,258])
        # plt.xticks([230,235,240,245,250,255])
        # ax.set_xticklabels(['18 Aug','23 Aug','28 Aug','2 Sep','7 Sep','12 Sep'])
        plt.xlabel('Date')
        plt.ylabel('Z [km]')
        plt.ylim([0,9000])
        plt.yticks(axmajor)
        ax.set_yticklabels([0,3,6,9])
        ax.set_yticks(axminor, minor = True)
        cbaxes = fig.add_axes([0.85, 0.12, 0.015, 0.3])
        cb = plt.colorbar(img, cax = cbaxes, orientation = 'vertical')
        plt.ylabel('N$_{aer, sol, coarse}$ [cm$^{-3}$]', rotation = 270, labelpad = 25)

        plt.show()


        data['monc']['q_accum_sol_number'] = np.zeros(arrlen)


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

def thetaTendencies(data):

    '''
    Design theta tendency input from subsequent two sondes
        f_force_pl_th, z_force_pl_th
    '''

    print ('Designing theta tendency input from subsequent two sondes:')

    data['thTend_flag'] = 1

    ### plot sonde theta profiles to check data has loaded correctly

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
    # plt.figure(figsize=(6,5))
    # plt.rc('legend',fontsize=MED_SIZE)
    # plt.subplots_adjust(top = 0.9, bottom = 0.15, right = 0.95, left = 0.15,
    #         hspace = 0.22, wspace = 0.5)
    #
    # plt.plot(data['sonde']['pottemp'][:] + 273.16, data['sonde']['Z'], label = 'SONDE')
    # plt.plot(data['sonde+1']['pottemp'][:] + 273.16, data['sonde+1']['Z'], label = 'SONDE+1')
    # plt.plot(data['sonde+2']['pottemp'][:] + 273.16, data['sonde+2']['Z'], label = 'SONDE+2')
    # plt.plot(data['sonde+3']['pottemp'][:] + 273.16, data['sonde+3']['Z'], label = 'SONDE+3')
    # plt.ylim([0,2.5e3])
    # plt.xlim([265,290])
    # plt.legend()
    # plt.ylabel('Z [m]')
    # plt.xlabel('$\Theta$ [K]')
    # plt.show()

    ####    ---------------
    ### want to calculate theta tendency (in K/day) between sonde0 and sondeX
    X = 1
    data['monc']['sondeX'] = X

    data['sonde' + str(X) + '-sonde0'] = {}

    ## change to give K/day
        ### if X = 2, 4/2 = 2 (12h)
        ### if X = 1, 4/1 = 4 (6h)
    data['sonde' + str(X) + '-sonde0']['th'] = (data['sonde+' + str(X)]['pottemp'] - data['sonde']['pottemp'])*(4./X)

    ####    ---------------
    ### want to regrid theta tendency (in K/day) to monc vertical grid

    ### build thTend array
    data['monc']['thTend'] = np.zeros(np.size(data['monc']['z']))
    interp_thTend = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde' + str(X) + '-sonde0']['th'][:])
    data['monc']['thTend'][1:] = interp_thTend(data['monc']['z'][1:])
    data['monc']['thTend'][0] = data['sonde' + str(X) + '-sonde0']['th'][0]

    ### build thRelax array
    data['monc']['thRelax'] = np.zeros(np.size(data['monc']['z']))
    interp_thRelax = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde+' + str(X)]['pottemp'][:]+273.16)
    data['monc']['thRelax'][1:] = interp_thRelax(data['monc']['z'][1:])
    data['monc']['thRelax'][0] = data['sonde+' + str(X)]['pottemp'][0]+273.16

    data['monc']['thref'] = np.zeros(np.size(data['monc']['z']))
    data['monc']['thref'][:] = 267.17
    data['monc']['threfRelax'] = data['monc']['thRelax'] - data['monc']['thref']

    ####    --------------- FIGURE

    print (data['monc']['z'].shape)
    print (data['monc']['thTend'].shape)
    print (data['monc']['thRelax'].shape)

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(9,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.15, right = 0.95, left = 0.15,
            hspace = 0.22, wspace = 0.5)

    plt.subplot(121)
    plt.plot(data['sonde']['pottemp'][:] + 273.16, data['sonde']['Z'], label = 'SONDE')
    plt.plot(data['monc']['thinit'], data['monc']['z'][:], 'k.', label = 'monc-thinit')
    plt.plot(data['sonde+' + str(X)]['pottemp'][:] + 273.16, data['sonde+2']['Z'], label = 'SONDE+' + str(X))
    plt.plot(data['monc']['thRelax'], data['monc']['z'][:], 'ks', markersize = 3, label = 'monc-thRelax')
    plt.ylim([0,2.5e3])
    plt.xlim([265,290])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('$\Theta$ [K]')

    plt.subplot(122)
    plt.plot([0,0],[0,2.5e3],'--', color = 'lightgrey')
    plt.plot(data['sonde' + str(X) + '-sonde0']['th'], data['sonde']['Z'], label = 'SONDE' + str(X) + '-SONDE0')
    plt.plot(data['monc']['thTend'], data['monc']['z'][:], 'kd', markersize = 4, label = 'monc-thTend')
    plt.plot(data['monc']['threfRelax'], data['monc']['z'][:], 'ks', markersize = 3, label = 'monc-thRelax')
    plt.ylim([0,2.5e3])
    # plt.xlim([265,290])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('$\Delta \Theta$ [K day$^{-1}$]')
    plt.savefig('../MOCCHA/FIGS/' + data['sonde_option'] + '-sonde' + str(X) + '_thetaTendency_thetaRelax.png')
    plt.show()



    return data

def qvTendencies(data):

    '''
    Design theta tendency input from subsequent two sondes
        f_force_pl_th, z_force_pl_th
    '''

    print ('Designing theta tendency input from subsequent two sondes:')

    data['qvTend_flag'] = 1

    ### plot sonde theta profiles to check data has loaded correctly

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
    # plt.figure(figsize=(6,5))
    # plt.rc('legend',fontsize=MED_SIZE)
    # plt.subplots_adjust(top = 0.9, bottom = 0.15, right = 0.95, left = 0.15,
    #         hspace = 0.22, wspace = 0.5)
    #
    # plt.plot(data['sonde']['pottemp'][:] + 273.16, data['sonde']['Z'], label = 'SONDE')
    # plt.plot(data['sonde+1']['pottemp'][:] + 273.16, data['sonde+1']['Z'], label = 'SONDE+1')
    # plt.plot(data['sonde+2']['pottemp'][:] + 273.16, data['sonde+2']['Z'], label = 'SONDE+2')
    # plt.plot(data['sonde+3']['pottemp'][:] + 273.16, data['sonde+3']['Z'], label = 'SONDE+3')
    # plt.ylim([0,2.5e3])
    # plt.xlim([265,290])
    # plt.legend()
    # plt.ylabel('Z [m]')
    # plt.xlabel('$\Theta$ [K]')
    # plt.show()

    ####    ---------------
    ### want to calculate qv tendency (in kg/kg/day) between sonde0 and sondeX

    if 'sondeX' in data.keys():
        print ('sonde' + str(X) + ' already chosen')
        X = data['monc']['sondeX']
    else:
        X = 1

    if 'sonde' + str(X) + '-sonde0' in data.keys():
        print ('sonde' + str(X) + '-sonde0 key already made')
    else:
        data['sonde' + str(X) + '-sonde0'] = {}

    ## change over 24 h (g/kg/day)
        ### if X = 2, 4/2 = 2 (12h)
        ### if X = 1, 4/1 = 4 (6h)
    data['sonde' + str(X) + '-sonde0']['qvapour'] = (data['sonde+' + str(X)]['sphum'] - data['sonde']['sphum'])*(X/4)

    ####    ---------------
    ### want to regrid qv tendency (in kg/kg/day) to monc vertical grid

    ### build thref array
    data['monc']['qvTend'] = np.zeros(np.size(data['monc']['z']))
    interp_qvTend = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde' + str(X) + '-sonde0']['qvapour'][:])
    data['monc']['qvTend'][1:] = interp_qvTend(data['monc']['z'][1:])/1e3
    data['monc']['qvTend'][0] = data['sonde' + str(X) + '-sonde0']['qvapour'][0]/1e3

    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(9,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.15, right = 0.95, left = 0.15,
            hspace = 0.22, wspace = 0.5)

    plt.subplot(121)
    plt.plot(data['sonde']['sphum'][:]/1e3, data['sonde']['Z'], label = 'SONDE')
    plt.plot(data['monc']['qinit1'], data['monc']['z'][:], 'k.', label = 'monc-namelist')
    plt.plot(data['sonde+' + str(X)]['sphum'][:]/1e3, data['sonde+' + str(X)]['Z'], label = 'SONDE+' + str(X))
    plt.ylim([0,2.5e3])
    # plt.xlim([265,290])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('q$_{v}$ [kg kg$^{-1}$]')

    plt.subplot(122)
    plt.plot([0,0],[0,2.5e3],'--', color = 'lightgrey')
    plt.plot(data['sonde' + str(X) + '-sonde0']['qvapour']/1e3, data['sonde']['Z'], label = 'SONDE' + str(X) + '-SONDE0')
    plt.plot(data['monc']['qvTend'], data['monc']['z'][:], 'kd', markersize = 4, label = 'monc-namelist')
    plt.ylim([0,2.5e3])
    # plt.xlim([265,290])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('$\Delta$ q$_{v}$ [kg kg$^{-1}$ day$^{-1}$]')
    plt.savefig('../MOCCHA/FIGS/' + data['sonde_option'] + '-sonde' + str(X) + '_qvTendency.png')
    plt.show()

    return data

def windTendencies(data):

    '''
    Design theta tendency input from subsequent two sondes
                f_force_pl_u, z_force_pl_u
                f_force_pl_v, z_force_pl_v
    '''

    print ('Designing wind tendency input from subsequent two sondes:')

    data['uvTend_flag'] = 1

    ### plot sonde theta profiles to check data has loaded correctly

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
    # plt.figure(figsize=(6,5))
    # plt.rc('legend',fontsize=MED_SIZE)
    # plt.subplots_adjust(top = 0.9, bottom = 0.15, right = 0.95, left = 0.15,
    #         hspace = 0.22, wspace = 0.5)
    #
    # plt.plot(data['sonde']['pottemp'][:] + 273.16, data['sonde']['Z'], label = 'SONDE')
    # plt.plot(data['sonde+1']['pottemp'][:] + 273.16, data['sonde+1']['Z'], label = 'SONDE+1')
    # plt.plot(data['sonde+2']['pottemp'][:] + 273.16, data['sonde+2']['Z'], label = 'SONDE+2')
    # plt.plot(data['sonde+3']['pottemp'][:] + 273.16, data['sonde+3']['Z'], label = 'SONDE+3')
    # plt.ylim([0,2.5e3])
    # plt.xlim([265,290])
    # plt.legend()
    # plt.ylabel('Z [m]')
    # plt.xlabel('$\Theta$ [K]')
    # plt.show()

    ####    ---------------
    ### want to calculate wind tendency (in K/day) between sonde0 and and sondeX

    if 'sondeX' in data.keys():
        print ('sonde' + str(X) + ' already chosen')
        X = data['monc']['sondeX']
    else:
        X = 1

    if 'sonde' + str(X) + '-sonde0' in data.keys():
        print ('sonde' + str(X) + '-sonde0 key already made')
    else:
        data['sonde' + str(X) + '-sonde0'] = {}

    ## change over 12 h (*2 to give m/s/day)
        ### if X = 2, 4/2 = 2 (12h)
        ### if X = 1, 4/1 = 4 (6h)
    data['sonde' + str(X) + '-sonde0']['u'] = (np.abs(data['sonde']['u']) - np.abs(data['sonde+' + str(X)]['u']))*(4./X)
    data['sonde' + str(X) + '-sonde0']['v'] = (np.abs(data['sonde']['v']) - np.abs(data['sonde+' + str(X)]['v']))*(4./X)

    data['sonde' + str(X) + '-sonde0']['v'][data['sonde' + str(X) + '-sonde0']['v'] > 1e3] = np.nan

    ####    ---------------
    ### want to regrid u/v tendency (in m/s/day) to monc vertical grid

    ### build uTend array
    data['monc']['uTend'] = np.zeros(np.size(data['monc']['z']))
    interp_uTend = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde' + str(X) + '-sonde0']['u'][:])
    data['monc']['uTend'][1:] = interp_uTend(data['monc']['z'][1:])
    data['monc']['uTend'][0] = data['sonde' + str(X) + '-sonde0']['u'][0]

    ### build uRelax array
    data['monc']['uRelax'] = np.zeros(np.size(data['monc']['z']))
    interp_uRelax = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde+' + str(X)]['u'][:])
    data['monc']['uRelax'][1:] = interp_uRelax(data['monc']['z'][1:])
    data['monc']['uRelax'][0] = data['sonde+' + str(X)]['u'][0]

    ### build vTend array
    data['monc']['vTend'] = np.zeros(np.size(data['monc']['z']))
    interp_vTend = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde' + str(X) + '-sonde0']['v'][:])
    data['monc']['vTend'][1:] = interp_vTend(data['monc']['z'][1:])
    data['monc']['vTend'][0] = data['sonde' + str(X) + '-sonde0']['v'][0]

    ### build vRelax array
    data['monc']['vRelax'] = np.zeros(np.size(data['monc']['z']))
    interp_vRelax = interp1d(np.squeeze(data['sonde']['Z'][:]),data['sonde+' + str(X)]['v'][:])
    data['monc']['vRelax'][1:] = interp_vRelax(data['monc']['z'][1:])
    data['monc']['vRelax'][0] = data['sonde+' + str(X)]['v'][0]



    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(9,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.15, right = 0.95, left = 0.15,
            hspace = 0.22, wspace = 0.5)

    plt.subplot(121)
    plt.plot([0,0],[0,2.5e3],'--', color = 'lightgrey')
    plt.plot(data['sonde']['u'][:], data['sonde']['Z'], label = 'SONDE')
    plt.plot(data['monc']['u'], data['monc']['z'][:], 'k.', label = 'monc-u')
    plt.plot(data['sonde+' + str(X)]['u'][:], data['sonde+' + str(X)]['Z'], label = 'SONDE+' + str(X))
    plt.plot(data['monc']['uRelax'], data['monc']['z'][:], 'ks',markersize = 3, label = 'monc-uRelax')
    plt.plot(data['monc']['uRelax'][::2], data['monc']['z'][::2], 'rs',markersize = 3, label = 'monc-uRelax[::2]')
    plt.ylim([0,2.5e3])
    plt.xlim([-20,10])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('u [m s$^{-1}$]')

    plt.subplot(122)
    plt.plot([0,0],[0,2.5e3],'--', color = 'lightgrey')
    plt.plot(data['sonde' + str(X) + '-sonde0']['u'], data['sonde']['Z'], label = 'SONDE' + str(X) + '-SONDE0')
    plt.plot(data['monc']['uTend'], data['monc']['z'][:], 'k.', label = 'monc-uTend')
    plt.ylim([0,2.5e3])
    # plt.xlim([-20,10])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('$\Delta$ u [m s$^{-1}$ day$^{-1}$]')
    plt.savefig('../MOCCHA/FIGS/' + data['sonde_option'] + '-sonde' + str(X) + '_uTendency.png')
    plt.show()


    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(9,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.15, right = 0.95, left = 0.15,
            hspace = 0.22, wspace = 0.5)

    plt.subplot(121)
    plt.plot([0,0],[0,2.5e3],'--', color = 'lightgrey')
    plt.plot(data['sonde']['v'][:], data['sonde']['Z'], label = 'SONDE')
    plt.plot(data['monc']['v'], data['monc']['z'][:], 'k.', label = 'monc-v')
    plt.plot(data['sonde+' + str(X)]['v'][:], data['sonde+' + str(X)]['Z'], label = 'SONDE+' + str(X))
    plt.plot(data['monc']['vRelax'], data['monc']['z'][:], 'ks', markersize = 3, label = 'monc-vRelax')
    plt.plot(data['monc']['vRelax'][::2], data['monc']['z'][::2], 'rs',markersize = 3, label = 'monc-vRelax[::2]')
    plt.ylim([0,2.5e3])
    plt.xlim([-20,10])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('v [m s$^{-1}$]')

    plt.subplot(122)
    plt.plot([0,0],[0,2.5e3],'--', color = 'lightgrey')
    plt.plot(data['sonde' + str(X) + '-sonde0']['v'], data['sonde']['Z'], label = 'SONDE' + str(X) + '-SONDE0')
    plt.plot(data['monc']['vTend'], data['monc']['z'][:], 'k.', label = 'monc-vTend')
    plt.ylim([0,2.5e3])
    # plt.xlim([-20,10])
    plt.legend()
    plt.ylabel('Z [m]')
    plt.xlabel('$\Delta$ v [m s$^{-1}$ day$^{-1}$]')
    plt.savefig('../MOCCHA/FIGS/' + data['sonde_option'] + '-sonde' + str(X) + '_vTendency.png')
    plt.show()


    return data

def moncInput(data):


        print ('***')
        print ('***')
        ### print out to terminal in format for monc namelists
        print ('z_init_pl_theta=')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_theta = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_theta=')
        for line in data['monc']['thinit']: sys.stdout.write('' + str(np.round(line,2)).strip() + ',')
        print ('')

                    # f_init_pl_theta = 267.27,267.58,267.59,267.57,267.65,267.74,268.06,268.91,269.5,269.95,270.35,270.53,270.72,271.07,271.46,272.2,273.29,273.68,274.34,274.53,274.62,275.2,276.39,276.7,276.77,276.89,279.95,282.59,282.9,283.5,284.72,286.12,287.18,287.81,288.38,288.95

        print ('z_init_pl_u=')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_u = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_u=')
        for line in data['monc']['u']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
        print ('')

                    # f_init_pl_u = -9.4125,-10.43,-11.875,-12.43,-12.55,-12.17,-11.38167,-10.9,-11.57,-12.7,-13.54,-13.45,-12.86,-11.93,-11.08333,-10.9,-11.28333,-11.95,-12.55,-12.7,-12.45,-10.83,-9.75,-9.6,-10.52333,-11.2,-11.2,-11.85,-11.7875,-11.28333,-10.96667,-11.35,-12.55,-12.03333,-11.55,-10.95

        print ('z_init_pl_v=')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_v = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_v=')
        for line in data['monc']['v']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
        print ('')

                    # f_init_pl_v = 0.825,1.74,2.85,2.9,3.26,3.93,4.4,4.11,3.03,2.5,3.2775,4.16,4.72,4.75,4.05,2.46667,1.08333,0.6,1.15,2.13167,2.85,2.93,0.11,0.01333,0.8,-0.40333,-3.175,-2.7,-3.0,-2.9,-2.95,-3.45,-3.0,-4.15,-4.45,-4.6

        print ('z_init_pl_q=')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_q = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0,2500.0

        print ('f_init_pl_q=')
        for line in data['monc']['qinit']: sys.stdout.write('' + str(np.round(line,10)).strip() + ',')
        print ('')

                    # f_init_pl_q = 0.00244,0.00227,0.00228,0.0023,0.00221,0.00214,0.00213,0.00215,0.00215,0.00212,0.00213,0.00211,0.00207,0.00203,0.00202,0.00199,0.00201,0.00201,0.00201,0.00204,0.00202,0.00198,0.00206,0.00206,0.00202,0.00201,0.00243,0.00197,0.00184,0.00184,0.00142,0.00107,0.00068,0.00062,0.00054,0.00042,0.0,0.0,0.0,0.0,6e-05,6e-05,6e-05,6e-05,6e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,

        if data['qAccum_flag'] == 1:
            print ('')
            print ('Reduced accumulation mode information for manual MONC restart:')
            print ('names_init_pl_q=accum_sol_mass, accum_sol_number')
            print ('z_init_pl_q=')
            for line in data['monc']['z'][::10]: sys.stdout.write('' + str(line).strip() + ',')
            print ('')
            print ('f_init_pl_q=')
            for line in data['monc']['q_accum_sol_mass'][::10]: sys.stdout.write('' + str(np.round(line,10)).strip() + ',')
            for line in data['monc']['q_accum_sol_number'][::10]: sys.stdout.write('' + str(np.round(line)).strip() + ',')
            # for line in data['monc']['thRelax']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
            print ('')
            print ('')

        if data['thTend_flag'] == 1:
            print ('z_force_pl_th=')
            for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
            print ('')
            print ('f_force_pl_th=')
            # for line in data['monc']['thTend']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
            for line in data['monc']['threfRelax']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
            print ('')

            print ('')
            print ('Reduced theta forcing for manual MONC restart:')
            print ('z_force_pl_th=')
            for line in data['monc']['z'][::2]: sys.stdout.write('' + str(line).strip() + ',')
            print ('')
            print ('f_force_pl_th=')
            # for line in data['monc']['thTend'][::2]: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
            for line in data['monc']['threfRelax'][::2]: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
            print ('')
            print ('')

        if data['qvTend_flag'] == 1:
            print ('z_force_pl_q=')
            for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
            print ('')
            print ('f_force_pl_q=')
            for line in data['monc']['qvTend']: sys.stdout.write('' + str(np.round(line,11)).strip() + ',')
            print ('')

        if data['uvTend_flag'] == 1:
            print ('z_force_pl_u=')
            for line in data['monc']['z'][::2]: sys.stdout.write('' + str(line).strip() + ',')
            print ('')
            print ('f_force_pl_u=')
            # for line in data['monc']['uTend']: sys.stdout.write('' + str(np.round(line,3)).strip() + ',')
            for line in data['monc']['uRelax'][::2]: sys.stdout.write('' + str(np.round(line,3)).strip() + ',')
            print ('')

            print ('z_force_pl_v=')
            for line in data['monc']['z'][::2]: sys.stdout.write('' + str(line).strip() + ',')
            print ('')
            print ('f_force_pl_v=')
            # for line in data['monc']['vTend']: sys.stdout.write('' + str(np.round(line,3)).strip() + ',')
            for line in data['monc']['vRelax'][::2]: sys.stdout.write('' + str(np.round(line,3)).strip() + ',')
            print ('')


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

    platform = 'LAPTOP'
    ####            options:
    ####                LAPTOP, JASMIN

    print ('Load radiosonde data from Jutta...')
    if platform == 'LAPTOP':
        obs_root_dir = '/home/gillian/MOCCHA/MOCCHA_GIT/ODEN/DATA/'
    elif platform == 'JASMIN':
        obs_root_dir = '/gws/nopw/j04/ncas_radar_vol1/gillian/Obs/'

    sondes = readMatlabStruct(obs_root_dir + 'radiosondes/SondeData_h10int_V02.mat')

    print ('')
    print (sondes.keys())

    ## -------------------------------------------------------------
    ## Choose sonde for initialisation:
    ## -------------------------------------------------------------
    data = {}
    data['sonde_option'] = '20180912T1800' # '20180912T1800' #'20180913T0000'#

    if data['sonde_option'] == '20180912T1800':
        numindex = 0
    elif data['sonde_option'] == '20180913T0000':
        numindex = 1
    elif data['sonde_option'] == '20180913T0600':
        numindex = 2

    ## -------------------------------------------------------------
    ## Load radiosonde (relative to 20180912 1200UTC
    ## -------------------------------------------------------------
    index256 = np.where(np.logical_or(np.round(sondes['doy'][:,:]) == 256., np.round(sondes['doy'][:,:]) == 257.))

    print (sondes['doy'][:,index256[1][numindex]])
    data['sonde'] = {}
    for k in sondes.keys():
        if k == 'Z': continue
        data['sonde'][k] = sondes[k][:,index256[1][numindex]]
    data['sonde']['Z'] = sondes['Z']
    data['sonde']['u'][data['sonde']['u'] > 1e3] = np.nan
    data['sonde']['v'][data['sonde']['v'] > 1e3] = np.nan

    ### load subsequent sondes
    for i in np.arange(0,3):
        data['sonde+' + str(i+1)] = {}
        print ('sonde+' + str(i+1))
        print (sondes['doy'][:,index256[1][i+1+numindex]])
        for k in sondes.keys():
            if k == 'Z': continue
            data['sonde+' + str(i+1)][k] = sondes[k][:,index256[1][i+1+numindex]]
        data['sonde+' + str(i+1)]['Z'] = sondes['Z']
        data['sonde+' + str(i+1)]['u'][data['sonde+' + str(i+1)]['u'] > 1e3] = np.nan
        data['sonde+' + str(i+1)]['v'][data['sonde+' + str(i+1)]['v'] > 1e3] = np.nan


    print (data['sonde'].keys())
    print (data['sonde']['doy'][:])

    ## -------------------------------------------------------------
    ## Initialise conditional flags for output
    ## -------------------------------------------------------------
    data['thTend_flag'] = 0     # theta tendencies?
    data['qvTend_flag'] = 0     # wind tendencies?
    data['uvTend_flag'] = 0     # wind tendencies?
    data['qAccum_flag'] = 0     # accumulation mode aerosol used?

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
    # data = aerosolACCUM(data)

    ### design tendency profiles
    ###     monc input will not be printed unless active
    # data = thetaTendencies(data)
    # data = qvTendencies(data)
    data = windTendencies(data)

    ## -------------------------------------------------------------
    ## Print out data in monc namelist format
    ## -------------------------------------------------------------
    data = moncInput(data)

    ## -------------------------------------------------------------
    ## save out working data for testing
    ## -------------------------------------------------------------
    # np.save('working_data', data)

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
