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
from physFuncts import calcThetaE, calcThetaVL, adiabatic_lwc, calcTemperature
from pyFixes import py3_FixNPLoad

def quicklooksSonde(data, sondenumber):

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

    yylim = 3e3

    print (data['sonde'].variables['theta'])
    print (data['sonde'].variables['theta'][:])

    # theta = data['sonde'].variables['theta'][:]
    # theta[theta.mask == True] = np.nan
    # print (theta)
    #
    # alt = data['sonde'].variables['alt'][:]
    # alt[alt.mask == True] = np.nan
    # print (alt)

    theta = np.arange(0.,11.,1.)
    alt = theta * 3.

    plt.subplot(131)
    plt.plot(theta, alt)
    plt.ylabel('Z [m]')
    plt.xlabel('Theta [K]')
    # plt.ylim([0,yylim])
    # plt.xlim([265,276])

    # plt.subplot(132)
    # plt.plot(data['sonde'].variables['mr'][:], data['sonde'].variables['alt'][:])
    # plt.xlabel('Mixing Ratio [g/kg]')
    # plt.ylim([0,yylim])
    #
    # plt.subplot(133)
    # plt.plot(data['sonde'].variables['rh'][:], data['sonde'].variables['alt'][:])
    # plt.xlabel('Rel. Hum. [%]')
    # plt.ylim([0,yylim])

    plt.savefig('../../../SHARE/Quicklooks_ACCACIA_B762-sonde5.png')
    plt.close()

def LEM_LoadTHREF(data, sondenumber):

    '''
    Load initialisation reference potential temperature profile
        -- Data copied from ncas_weather/gyoung/LEM/r143/nmlsetup
    '''
    #### dz is 5m between 0 and 1157.5 m, then linearly interpolated to 10m between 1285m and 2395m
    data['accacia'] = {}
    data['accacia']['z'] = np.array([6.8,49.9,91.9,134.9,181.3,233.1,287.4,341.7,396.5,449.4,502.6,
        565.8,625.3,682.2,728.4,777.8,828.7,880.5,932.4,984.7,1035.6,1084.6,1136.7,
        1190.2,1243.0,1302.5,1355.5,1407.0,1459.3,1510.7,1561.4,1611.7,1662.4,1712.7,
        1763.9,1814.6,1866.2,1917.7,1970.1,2022.5,2074.6,2127.6,2179.6,2231.6,2284.1,
        2336.7,2389.4,2442.4,2496.1,2549.5,2603.2,2656.8,2710.6,2764.6,2818.9,2884.2,2939.0,3005.0])

    data['accacia']['theta'] = np.array([262.79,262.64,262.55,262.64,262.43,262.36,262.39,262.42,262.46,
        262.50,262.59,262.71,262.83,262.93,263.02,263.11,263.20,263.26,263.30,263.40,263.52,263.60,263.80,
        264.53,266.08,267.63,268.70,269.24,269.51,269.97,270.44,271.00,271.43,271.77,272.06,272.26,272.38,
        272.50,272.65,272.85,273.03,273.19,273.46,273.97,274.68,275.20,275.55,275.83,276.20,276.56,276.91,
        277.23,277.49,277.86,278.27,278.54,278.63,278.75])

    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(4,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.25,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.4e3

    plt.plot(data['accacia']['theta'], data['accacia']['z'], label = 'ACCACIA')
    # plt.plot(data['theta'], data['z'], label = 'SONDE')
    plt.ylabel('Z [m]')
    plt.xlabel('$\Theta$ [K]')
    plt.ylim([0,yylim])
    # plt.xlim([265,295])

    plt.savefig('../../../SHARE/Quicklooks_ACCACIA_B762-sonde5-LEM_theta.png')
    plt.show()

    return data

def LEM_LoadWINDS(data, sondenumber):

    '''
    Load initialisation reference wind profiles
        -- Data copied from ncas_weather/gyoung/LEM/r143/nmlsetup
    '''
    data['accacia']['z'] = np.array([6.8,49.9,91.9,134.9,181.3,233.1,287.4,341.7,396.5,449.4,502.6,
        565.8,625.3,682.2,728.4,777.8,828.7,880.5,932.4,984.7,1035.6,1084.6,1136.7,
        1190.2,1243.0,1302.5,1355.5,1407.0,1459.3,1510.7,1561.4,1611.7,1662.4,1712.7,
        1763.9,1814.6,1866.2,1917.7,1970.1,2022.5,2074.6,2127.6,2179.6,2231.6,2284.1,
        2336.7,2389.4,2442.4,2496.1,2549.5,2603.2,2656.8,2710.6,2764.6,2818.9,2884.2,2939.0,3005.0])

    data['accacia']['v'] = np.array([-8.66,-9.20,-10.22,-9.66,-9.66,-8.72,-8.25,-8.50,-9.19,-9.35,-9.29,
        -9.57,-9.99,-10.64,-11.30,-11.05,-11.19,-10.79,-10.91,-10.59,-9.85,-10.33,-10.17,-10.81,-12.56,
        -14.70,-15.78,-15.78,-16.23,-16.16,-15.68,-15.17,-14.90,-15.09,-15.27,-14.76,-14.32,-13.67,-13.41,
        -13.18,-12.92,-12.92,-12.88,-13.57,-13.99,-13.75,-13.83,-14.94,-15.64,-16.24,-16.18,-16.27,-16.98,
        -16.76,-16.28,-15.83,-16.09,-16.50])

    data['accacia']['u'] = np.array([0.22,-0.41,-0.46,-1.62,-1.14,-0.67,-1.68,-1.23,-0.65,-0.89,-1.25,-0.84,
    -1.20,-0.82,-1.60,-2.14,-1.74,-1.72,-2.31,-2.68,-2.53,-2.47,-2.77,-1.76,-0.99,-0.74,-0.42,-0.94,-1.12,
    -0.52,-1.08,-1.77,-1.33,0.36,1.29,1.15,1.15,0.87,1.03,1.41,2.24,2.67,3.00,3.07,2.85,2.67,2.42,2.85,
    2.57,3.12,3.92,3.54,3.97,4.04,4.09,3.60,3.04,2.60])

    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(4,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.25,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.4e3

    plt.plot(data['accacia']['u'], data['accacia']['z'], label = 'U')
    plt.plot(data['accacia']['v'], data['accacia']['z'], label = 'V')
    # plt.plot(data['theta'], data['z'], label = 'SONDE')
    plt.ylabel('Z [m]')
    plt.xlabel('Wind speed [m/s]')
    plt.ylim([0,yylim])
    plt.legend()
    # plt.xlim([265,295])

    plt.savefig('../../../SHARE/Quicklooks_ACCACIA_B762-sonde5-LEM_winds.png')
    plt.show()

    return data

def LEM_LoadQ01(data,sondenumber):

    '''
    Load initialisation reference wind profiles
        -- Data copied from ncas_weather/gyoung/LEM/r143/nmlsetup
    '''
    data['accacia']['z'] = np.array([6.8,49.9,91.9,134.9,181.3,233.1,287.4,341.7,396.5,449.4,502.6,
        565.8,625.3,682.2,728.4,777.8,828.7,880.5,932.4,984.7,1035.6,1084.6,1136.7,
        1190.2,1243.0,1302.5,1355.5,1407.0,1459.3,1510.7,1561.4,1611.7,1662.4,1712.7,
        1763.9,1814.6,1866.2,1917.7,1970.1,2022.5,2074.6,2127.6,2179.6,2231.6,2284.1,
        2336.7,2389.4,2442.4,2496.1,2549.5,2603.2,2656.8,2710.6,2764.6,2818.9,2884.2,2939.0,3005.0])

    data['accacia']['q01'] = np.array([1.601e-03,1.574e-03,1.509e-03,1.530e-03,1.461e-03,
        1.408e-03,1.379e-03,1.350e-03,1.318e-03,1.297e-03,1.278e-03,1.253e-03,1.225e-03,
        1.196e-03,1.178e-03,1.157e-03,1.137e-03,1.113e-03,1.077e-03,1.046e-03,1.018e-03,
        9.777e-04,9.293e-04,9.085e-04,9.792e-04,1.012e-03,1.050e-03,1.099e-03,1.043e-03,
        9.399e-04,8.956e-04,8.601e-04,8.872e-04,9.279e-04,9.358e-04,9.443e-04,9.427e-04,
        9.390e-04,9.093e-04,8.918e-04,8.810e-04,8.617e-04,8.162e-04,7.527e-04,6.669e-04,
        6.832e-04,5.299e-04,4.755e-04,4.308e-04,4.338e-04,4.513e-04,4.846e-04,5.634e-04,
        5.923e-04,5.858e-04,5.952e-04,6.030e-04,6.070e-04])

    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(4,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.25,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.4e3

    plt.plot(data['accacia']['q01'], data['accacia']['z'], label = 'Q01')
    plt.ylabel('Z [m]')
    plt.xlabel('Q01 [kg/kg]')
    plt.ylim([0,yylim])
    plt.legend()
    # plt.xlim([265,295])

    plt.savefig('../../../SHARE/Quicklooks_ACCACIA_B762-sonde5-LEM_q01.png')
    plt.show()

    return data

def LEM_LoadQ02(data,sondenumber):

    '''
    Load initialisation reference Q02 profile
        -- Data copied from ncas_weather/gyoung/LEM/r143/nmlsetup
    '''
    data['accacia']['z'] = np.array([6.8,49.9,91.9,134.9,181.3,233.1,287.4,341.7,396.5,449.4,502.6,
        565.8,625.3,682.2,728.4,777.8,828.7,880.5,932.4,984.7,1035.6,1084.6,1136.7,
        1190.2,1243.0,1302.5,1355.5,1407.0,1459.3,1510.7,1561.4,1611.7,1662.4,1712.7,
        1763.9,1814.6,1866.2,1917.7,1970.1,2022.5,2074.6,2127.6,2179.6,2231.6,2284.1,
        2336.7,2389.4,2442.4,2496.1,2549.5,2603.2,2656.8,2710.6,2764.6,2818.9,2884.2,2939.0,3005.0])

    data['accacia']['q02'] = np.array([0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
        0.000e+00,1.124e-05,5.584e-05,9.825e-05,1.401e-04,1.889e-04,2.339e-04,2.760e-04,3.097e-04,
        3.450e-04,3.807e-04,4.164e-04,4.516e-04,4.862e-04,5.193e-04,5.505e-04,5.832e-04,0.000e+00,
        0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
        0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
        0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
        0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00])


    ####    --------------- FIGURE

    SMALL_SIZE = 12
    MED_SIZE = 14
    LARGE_SIZE = 16

    plt.rc('font',size=MED_SIZE)
    plt.rc('axes',titlesize=MED_SIZE)
    plt.rc('axes',labelsize=MED_SIZE)
    plt.rc('xtick',labelsize=MED_SIZE)
    plt.rc('ytick',labelsize=MED_SIZE)
    plt.figure(figsize=(4,5))
    plt.rc('legend',fontsize=MED_SIZE)
    plt.subplots_adjust(top = 0.9, bottom = 0.12, right = 0.95, left = 0.25,
            hspace = 0.22, wspace = 0.4)

    yylim = 2.4e3

    plt.plot(data['accacia']['q02'], data['accacia']['z'], label = 'Q01')
    plt.ylabel('Z [m]')
    plt.xlabel('Q02 [kg/kg]')
    plt.ylim([0,yylim])
    plt.legend()
    # plt.xlim([265,295])

    plt.savefig('../../../SHARE/Quicklooks_ACCACIA_B762-sonde5-LEM_q02.png')
    plt.show()

    return data


def aerosolACCUM(data):

    '''
    Design accummulation mode aerosol inputs:
        names_init_pl_q=accum_sol_mass, accum_sol_number
    '''

    print ('Designing soluble accummulation mode input:')

    # data['qAccum_flag'] = 1

    arrlen = np.size(data['monc']['z'])
    print(arrlen)

    case = 'CASIM-100'
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

    plt.savefig('../ASCOS/FIGS/NsolAccum_MsolAccum_' + case + '.png')
    plt.show()


    ### combine to existing q field input
    data['monc']['q_accum_sol'] = np.append(data['monc']['q_accum_sol_mass'], data['monc']['q_accum_sol_number'])
    data['monc']['qinit'] = np.append(data['monc']['qinit'], data['monc']['q_accum_sol'])


    return data

def moncInput(data, sondenumber):

        ### print out to terminal in format for monc namelists
        print ('z_init_pl_theta = ')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_theta = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,800.0,900.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0

        print ('f_init_pl_theta = ')
        for line in data['monc']['thref']: sys.stdout.write('' + str(np.round(line,2)).strip() + ',')
        print ('')

                    # f_init_pl_theta = 269.2,268.95,268.86,268.89,269.07,269.79,270.22,270.32,270.43,270.58,270.78,270.89,271.14,272.69,274.75,276.55,278.79,280.87,282.13,283.23,284.71,285.68,286.69,287.61,287.98,288.52,289.12,289.89,290.02,290.5,290.95,291.24

        print ('z_init_pl_q = ')
        for line in data['monc']['z']: sys.stdout.write('' + str(line).strip() + ',')
        print ('')

                    # z_init_pl_q = 0.0,50.0,100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,800.0,900.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2100.0,2200.0,2300.0,2400.0

        print ('f_init_pl_q = ')
        for line in data['monc']['qinit']: sys.stdout.write('' + str(np.round(line,5)).strip() + ',')
        print ('')

                    # f_init_pl_q = 0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00257,0.00246,0.00228,0.00211,0.00193,0.00176,0.00159,0.00141,0.00124,0.00105,0.00105,0.00105,0.00105,0.00105,0.00105,0.00105,0.00105,0.00105,0.00105,0.0,0.0,0.0,0.0,0.0,0.0,7e-05,7e-05,7e-05,7e-05,7e-05,7e-05,7e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

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

    plt.close() ### in case there are any trailing plot instances

    platform = 'JASMIN'

    # if platform == 'MAC':
    #     import matplotlib
    #     import glob
    #     matplotlib.use('TkAgg')
    #
    # import matplotlib.pyplot as plt



    print ('Import ACCACIA radiosonde data:')
    print ('...')

    # print ('Load radiosonde data...')
    # sondes = readMatlabData('../DATA/radiosondes.mat')

    print ('')
    # print (sondes.keys())

    ## -------------------------------------------------------------
    ## Load radiosonde from 20130323 0939 UTC
    ## -------------------------------------------------------------
    data = {}
    if platform == 'MAC':
        sondenumber = '/Users/eargy/KRAKENshare/faam-dropsonde_faam_20130323093914_r0_b762_proc.nc'
    elif platform == 'JASMIN':
        sondenumber = '/gws/nopw/j04/ncas_weather/gyoung/ACCACIA/CORE_FAAM/B762/radiosondes/faam-dropsonde_faam_20130323093914_r0_b762_proc.nc'

    data['sonde'] = Dataset(sondenumber,'r')

    ## -------------------------------------------------------------
    ## Quicklook plots of chosen sonde
    ## -------------------------------------------------------------
    # figure = quicklooksSonde(data, sondenumber)

    ## -------------------------------------------------------------
    ## Read in data from LEM namelists
    ## -------------------------------------------------------------
    data = LEM_LoadTHREF(data, sondenumber)
    data = LEM_LoadWINDS(data, sondenumber)
    data = LEM_LoadQ01(data, sondenumber)
    data = LEM_LoadQ02(data, sondenumber)

    print (data['accacia'].keys())

    ## -------------------------------------------------------------
    ## Aerosol inputs
    ## -------------------------------------------------------------
    # data = aerosolACCUM(data)

    ## -------------------------------------------------------------
    ## Print out data in monc namelist format
    ## -------------------------------------------------------------
    # data = moncInput(data, sondenumber)

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
