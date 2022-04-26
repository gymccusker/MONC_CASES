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

def LEM_LoadQ01(data,sondenumber):

    '''
    Load initialisation reference Q02 profile
        -- Data copied from ncas_weather/gyoung/LEM/r143/nmlsetup
    '''
    data['accacia']['z'] = np.array([6.8,49.9,91.9,134.9,181.3,233.1,287.4,341.7,396.5,449.4,502.6,
        565.8,625.3,682.2,728.4,777.8,828.7,880.5,932.4,984.7,1035.6,1084.6,1136.7,
        1190.2,1243.0,1302.5,1355.5,1407.0,1459.3,1510.7,1561.4,1611.7,1662.4,1712.7,
        1763.9,1814.6,1866.2,1917.7,1970.1,2022.5,2074.6,2127.6,2179.6,2231.6,2284.1,
        2336.7,2389.4,2442.4,2496.1,2549.5,2603.2,2656.8,2710.6,2764.6,2818.9,2884.2,2939.0,3005.0])

    data['accacia']['q02'] = np.array([1.601e-03,1.574e-03,1.509e-03,1.530e-03,1.461e-03,
        1.408e-03,1.379e-03,1.350e-03,1.318e-03,1.297e-03,1.278e-03,1.253e-03,1.225e-03,
        1.196e-03,1.178e-03,1.157e-03,1.137e-03,1.113e-03,1.077e-03,1.046e-03,1.018e-03,
        9.777e-04,9.293e-04,9.085e-04,9.792e-04,1.012e-03,1.050e-03,1.099e-03,1.043e-03,
        9.399e-04,8.956e-04,8.601e-04,8.872e-04,9.279e-04,9.358e-04,9.443e-04,9.427e-04,
        9.390e-04,9.093e-04,8.918e-04,8.810e-04,8.617e-04,8.162e-04,7.527e-04,6.669e-04,
        6.832e-04,5.299e-04,4.755e-04,4.308e-04,4.338e-04,4.513e-04,4.846e-04,5.634e-04,
        5.923e-04,5.858e-04,5.952e-04,6.030e-04,6.070e-04])

znqi_read(1,2)=6.8,qinit_read(1,2)=0.000e+00,
znqi_read(2,2)=11.0,qinit_read(2,2)=0.000e+00,
znqi_read(3,2)=15.3,qinit_read(3,2)=0.000e+00,
znqi_read(4,2)=19.6,qinit_read(4,2)=0.000e+00,
znqi_read(5,2)=23.9,qinit_read(5,2)=0.000e+00,
znqi_read(6,2)=28.2,qinit_read(6,2)=0.000e+00,
znqi_read(7,2)=32.6,qinit_read(7,2)=0.000e+00,
znqi_read(8,2)=36.9,qinit_read(8,2)=0.000e+00,
znqi_read(9,2)=41.3,qinit_read(9,2)=0.000e+00,
znqi_read(10,2)=49.9,qinit_read(10,2)=0.000e+00,
znqi_read(11,2)=54.1,qinit_read(11,2)=0.000e+00,
znqi_read(12,2)=58.4,qinit_read(12,2)=0.000e+00,
znqi_read(13,2)=62.6,qinit_read(13,2)=0.000e+00,
znqi_read(14,2)=66.8,qinit_read(14,2)=0.000e+00,
znqi_read(15,2)=71.0,qinit_read(15,2)=0.000e+00,
znqi_read(16,2)=75.2,qinit_read(16,2)=0.000e+00,
znqi_read(17,2)=79.4,qinit_read(17,2)=0.000e+00,
znqi_read(18,2)=83.6,qinit_read(18,2)=0.000e+00,
znqi_read(19,2)=87.8,qinit_read(19,2)=0.000e+00,
znqi_read(20,2)=91.9,qinit_read(20,2)=0.000e+00,
znqi_read(21,2)=96.0,qinit_read(21,2)=0.000e+00,
znqi_read(22,2)=100.1,qinit_read(22,2)=0.000e+00,
znqi_read(23,2)=104.2,qinit_read(23,2)=0.000e+00,
znqi_read(24,2)=108.2,qinit_read(24,2)=0.000e+00,
znqi_read(25,2)=116.1,qinit_read(25,2)=0.000e+00,
znqi_read(26,2)=119.9,qinit_read(26,2)=0.000e+00,
znqi_read(27,2)=123.7,qinit_read(27,2)=0.000e+00,
znqi_read(28,2)=127.4,qinit_read(28,2)=0.000e+00,
znqi_read(29,2)=131.2,qinit_read(29,2)=0.000e+00,
znqi_read(30,2)=134.9,qinit_read(30,2)=0.000e+00,
znqi_read(31,2)=138.7,qinit_read(31,2)=0.000e+00,
znqi_read(32,2)=142.6,qinit_read(32,2)=0.000e+00,
znqi_read(33,2)=146.5,qinit_read(33,2)=0.000e+00,
znqi_read(34,2)=154.7,qinit_read(34,2)=0.000e+00,
znqi_read(35,2)=158.9,qinit_read(35,2)=0.000e+00,
znqi_read(36,2)=163.3,qinit_read(36,2)=0.000e+00,
znqi_read(37,2)=167.6,qinit_read(37,2)=0.000e+00,
znqi_read(38,2)=172.1,qinit_read(38,2)=0.000e+00,
znqi_read(39,2)=176.6,qinit_read(39,2)=0.000e+00,
znqi_read(40,2)=181.3,qinit_read(40,2)=0.000e+00,
znqi_read(41,2)=186.0,qinit_read(41,2)=0.000e+00,
znqi_read(42,2)=190.9,qinit_read(42,2)=0.000e+00,
znqi_read(43,2)=195.9,qinit_read(43,2)=0.000e+00,
znqi_read(44,2)=201.0,qinit_read(44,2)=0.000e+00,
znqi_read(45,2)=206.2,qinit_read(45,2)=0.000e+00,
znqi_read(46,2)=211.5,qinit_read(46,2)=0.000e+00,
znqi_read(47,2)=216.8,qinit_read(47,2)=0.000e+00,
znqi_read(48,2)=222.2,qinit_read(48,2)=0.000e+00,
znqi_read(49,2)=227.7,qinit_read(49,2)=0.000e+00,
znqi_read(50,2)=233.1,qinit_read(50,2)=0.000e+00,
znqi_read(51,2)=238.6,qinit_read(51,2)=0.000e+00,
znqi_read(52,2)=244.1,qinit_read(52,2)=0.000e+00,
znqi_read(53,2)=249.5,qinit_read(53,2)=0.000e+00,
znqi_read(54,2)=254.9,qinit_read(54,2)=0.000e+00,
znqi_read(55,2)=260.4,qinit_read(55,2)=0.000e+00,
znqi_read(56,2)=265.8,qinit_read(56,2)=0.000e+00,
znqi_read(57,2)=271.2,qinit_read(57,2)=0.000e+00,
znqi_read(58,2)=276.6,qinit_read(58,2)=0.000e+00,
znqi_read(59,2)=282.0,qinit_read(59,2)=0.000e+00,
znqi_read(60,2)=287.4,qinit_read(60,2)=0.000e+00,
znqi_read(61,2)=292.8,qinit_read(61,2)=0.000e+00,
znqi_read(62,2)=298.2,qinit_read(62,2)=0.000e+00,
znqi_read(63,2)=303.6,qinit_read(63,2)=0.000e+00,
znqi_read(64,2)=309.0,qinit_read(64,2)=0.000e+00,
znqi_read(65,2)=314.4,qinit_read(65,2)=0.000e+00,
znqi_read(66,2)=319.9,qinit_read(66,2)=0.000e+00,
znqi_read(67,2)=325.3,qinit_read(67,2)=8.416e-07,
znqi_read(68,2)=330.8,qinit_read(68,2)=2.207e-06,
znqi_read(69,2)=336.2,qinit_read(69,2)=6.715e-06,
znqi_read(70,2)=341.7,qinit_read(70,2)=1.124e-05,
znqi_read(71,2)=347.3,qinit_read(71,2)=1.579e-05,
znqi_read(72,2)=352.8,qinit_read(72,2)=2.035e-05,
znqi_read(73,2)=358.4,qinit_read(73,2)=2.492e-05,
znqi_read(74,2)=364.0,qinit_read(74,2)=2.947e-05,
znqi_read(75,2)=369.5,qinit_read(75,2)=3.398e-05,
znqi_read(76,2)=375.0,qinit_read(76,2)=3.843e-05,
znqi_read(77,2)=380.4,qinit_read(77,2)=4.284e-05,
znqi_read(78,2)=385.8,qinit_read(78,2)=4.720e-05,
znqi_read(79,2)=391.1,qinit_read(79,2)=5.154e-05,
znqi_read(80,2)=396.5,qinit_read(80,2)=5.584e-05,
znqi_read(81,2)=401.8,qinit_read(81,2)=6.014e-05,
znqi_read(82,2)=407.1,qinit_read(82,2)=6.441e-05,
znqi_read(83,2)=412.4,qinit_read(83,2)=6.868e-05,
znqi_read(84,2)=417.7,qinit_read(84,2)=7.292e-05,
znqi_read(85,2)=423.0,qinit_read(85,2)=7.717e-05,
znqi_read(86,2)=428.3,qinit_read(86,2)=8.141e-05,
znqi_read(87,2)=433.6,qinit_read(87,2)=8.564e-05,
znqi_read(88,2)=438.9,qinit_read(88,2)=8.985e-05,
znqi_read(89,2)=444.2,qinit_read(89,2)=9.406e-05,
znqi_read(90,2)=449.4,qinit_read(90,2)=9.825e-05,
znqi_read(91,2)=454.7,qinit_read(91,2)=1.024e-04,
znqi_read(92,2)=460.0,qinit_read(92,2)=1.066e-04,
znqi_read(93,2)=465.2,qinit_read(93,2)=1.108e-04,
znqi_read(94,2)=470.5,qinit_read(94,2)=1.149e-04,
znqi_read(95,2)=475.8,qinit_read(95,2)=1.191e-04,
znqi_read(96,2)=481.1,qinit_read(96,2)=1.233e-04,
znqi_read(97,2)=486.5,qinit_read(97,2)=1.275e-04,
znqi_read(98,2)=491.8,qinit_read(98,2)=1.317e-04,
znqi_read(99,2)=497.2,qinit_read(99,2)=1.359e-04,
znqi_read(100,2)=502.6,qinit_read(100,2)=1.401e-04,
znqi_read(101,2)=507.9,qinit_read(101,2)=1.442e-04,
znqi_read(102,2)=513.3,qinit_read(102,2)=1.484e-04,
znqi_read(103,2)=518.6,qinit_read(103,2)=1.525e-04,
znqi_read(104,2)=523.9,qinit_read(104,2)=1.567e-04,
znqi_read(105,2)=529.3,qinit_read(105,2)=1.608e-04,
znqi_read(106,2)=534.5,qinit_read(106,2)=1.649e-04,
znqi_read(107,2)=550.2,qinit_read(107,2)=1.770e-04,
znqi_read(108,2)=555.4,qinit_read(108,2)=1.809e-04,
znqi_read(109,2)=560.6,qinit_read(109,2)=1.849e-04,
znqi_read(110,2)=565.8,qinit_read(110,2)=1.889e-04,
znqi_read(111,2)=571.1,qinit_read(111,2)=1.929e-04,
znqi_read(112,2)=576.4,qinit_read(112,2)=1.969e-04,
znqi_read(113,2)=581.7,qinit_read(113,2)=2.010e-04,
znqi_read(114,2)=587.0,qinit_read(114,2)=2.050e-04,
znqi_read(115,2)=592.4,qinit_read(115,2)=2.091e-04,
znqi_read(116,2)=597.9,qinit_read(116,2)=2.132e-04,
znqi_read(117,2)=603.3,qinit_read(117,2)=2.173e-04,
znqi_read(118,2)=608.8,qinit_read(118,2)=2.215e-04,
znqi_read(119,2)=614.3,qinit_read(119,2)=2.256e-04,
znqi_read(120,2)=625.3,qinit_read(120,2)=2.339e-04,
znqi_read(121,2)=630.9,qinit_read(121,2)=2.380e-04,
znqi_read(122,2)=636.3,qinit_read(122,2)=2.421e-04,
znqi_read(123,2)=641.8,qinit_read(123,2)=2.462e-04,
znqi_read(124,2)=647.2,qinit_read(124,2)=2.502e-04,
znqi_read(125,2)=652.5,qinit_read(125,2)=2.541e-04,
znqi_read(126,2)=662.8,qinit_read(126,2)=2.617e-04,
znqi_read(127,2)=667.8,qinit_read(127,2)=2.654e-04,
znqi_read(128,2)=672.7,qinit_read(128,2)=2.690e-04,
znqi_read(129,2)=677.5,qinit_read(129,2)=2.726e-04,
znqi_read(130,2)=682.2,qinit_read(130,2)=2.760e-04,
znqi_read(131,2)=686.9,qinit_read(131,2)=2.795e-04,
znqi_read(132,2)=691.6,qinit_read(132,2)=2.829e-04,
znqi_read(133,2)=696.2,qinit_read(133,2)=2.863e-04,
znqi_read(134,2)=700.8,qinit_read(134,2)=2.896e-04,
znqi_read(135,2)=705.4,qinit_read(135,2)=2.929e-04,
znqi_read(136,2)=709.9,qinit_read(136,2)=2.963e-04,
znqi_read(137,2)=714.5,qinit_read(137,2)=2.996e-04,
znqi_read(138,2)=719.1,qinit_read(138,2)=3.029e-04,
znqi_read(139,2)=723.7,qinit_read(139,2)=3.063e-04,
znqi_read(140,2)=728.4,qinit_read(140,2)=3.097e-04,
znqi_read(141,2)=733.2,qinit_read(141,2)=3.131e-04,
znqi_read(142,2)=738.0,qinit_read(142,2)=3.165e-04,
znqi_read(143,2)=742.8,qinit_read(143,2)=3.200e-04,
znqi_read(144,2)=747.8,qinit_read(144,2)=3.236e-04,
znqi_read(145,2)=752.7,qinit_read(145,2)=3.271e-04,
znqi_read(146,2)=757.7,qinit_read(146,2)=3.307e-04,
znqi_read(147,2)=762.7,qinit_read(147,2)=3.342e-04,
znqi_read(148,2)=767.7,qinit_read(148,2)=3.378e-04,
znqi_read(149,2)=772.7,qinit_read(149,2)=3.414e-04,
znqi_read(150,2)=777.8,qinit_read(150,2)=3.450e-04,
znqi_read(151,2)=782.8,qinit_read(151,2)=3.485e-04,
znqi_read(152,2)=787.8,qinit_read(152,2)=3.521e-04,
znqi_read(153,2)=792.9,qinit_read(153,2)=3.556e-04,
znqi_read(154,2)=797.9,qinit_read(154,2)=3.592e-04,
znqi_read(155,2)=803.0,qinit_read(155,2)=3.628e-04,
znqi_read(156,2)=808.1,qinit_read(156,2)=3.664e-04,
znqi_read(157,2)=813.2,qinit_read(157,2)=3.699e-04,
znqi_read(158,2)=818.3,qinit_read(158,2)=3.735e-04,
znqi_read(159,2)=823.5,qinit_read(159,2)=3.771e-04,
znqi_read(160,2)=828.7,qinit_read(160,2)=3.807e-04,
znqi_read(161,2)=833.8,qinit_read(161,2)=3.843e-04,
znqi_read(162,2)=839.0,qinit_read(162,2)=3.879e-04,
znqi_read(163,2)=844.1,qinit_read(163,2)=3.914e-04,
znqi_read(164,2)=849.3,qinit_read(164,2)=3.950e-04,
znqi_read(165,2)=854.4,qinit_read(165,2)=3.986e-04,
znqi_read(166,2)=859.6,qinit_read(166,2)=4.021e-04,
znqi_read(167,2)=864.8,qinit_read(167,2)=4.057e-04,
znqi_read(168,2)=870.1,qinit_read(168,2)=4.093e-04,
znqi_read(169,2)=875.3,qinit_read(169,2)=4.129e-04,
znqi_read(170,2)=880.5,qinit_read(170,2)=4.164e-04,
znqi_read(171,2)=885.7,qinit_read(171,2)=4.200e-04,
znqi_read(172,2)=890.9,qinit_read(172,2)=4.235e-04,
znqi_read(173,2)=896.1,qinit_read(173,2)=4.270e-04,
znqi_read(174,2)=901.2,qinit_read(174,2)=4.305e-04,
znqi_read(175,2)=906.4,qinit_read(175,2)=4.340e-04,
znqi_read(176,2)=911.5,qinit_read(176,2)=4.375e-04,
znqi_read(177,2)=916.7,qinit_read(177,2)=4.410e-04,
znqi_read(178,2)=921.9,qinit_read(178,2)=4.445e-04,
znqi_read(179,2)=927.1,qinit_read(179,2)=4.480e-04,
znqi_read(180,2)=932.4,qinit_read(180,2)=4.516e-04,
znqi_read(181,2)=937.7,qinit_read(181,2)=4.551e-04,
znqi_read(182,2)=943.0,qinit_read(182,2)=4.586e-04,
znqi_read(183,2)=948.3,qinit_read(183,2)=4.621e-04,
znqi_read(184,2)=953.5,qinit_read(184,2)=4.656e-04,
znqi_read(185,2)=958.7,qinit_read(185,2)=4.691e-04,
znqi_read(186,2)=964.0,qinit_read(186,2)=4.726e-04,
znqi_read(187,2)=969.2,qinit_read(187,2)=4.760e-04,
znqi_read(188,2)=974.3,qinit_read(188,2)=4.794e-04,
znqi_read(189,2)=979.5,qinit_read(189,2)=4.828e-04,
znqi_read(190,2)=984.7,qinit_read(190,2)=4.862e-04,
znqi_read(191,2)=989.8,qinit_read(191,2)=4.896e-04,
znqi_read(192,2)=994.9,qinit_read(192,2)=4.929e-04,
znqi_read(193,2)=1000.0,qinit_read(193,2)=4.963e-04,
znqi_read(194,2)=1005.2,qinit_read(194,2)=4.996e-04,
znqi_read(195,2)=1010.3,qinit_read(195,2)=5.030e-04,
znqi_read(196,2)=1015.4,qinit_read(196,2)=5.063e-04,
znqi_read(197,2)=1020.5,qinit_read(197,2)=5.096e-04,
znqi_read(198,2)=1025.6,qinit_read(198,2)=5.129e-04,
znqi_read(199,2)=1030.6,qinit_read(199,2)=5.161e-04,
znqi_read(200,2)=1035.6,qinit_read(200,2)=5.193e-04,
znqi_read(201,2)=1040.6,qinit_read(201,2)=5.225e-04,
znqi_read(202,2)=1045.5,qinit_read(202,2)=5.257e-04,
znqi_read(203,2)=1050.5,qinit_read(203,2)=5.289e-04,
znqi_read(204,2)=1055.3,qinit_read(204,2)=5.320e-04,
znqi_read(205,2)=1060.2,qinit_read(205,2)=5.351e-04,
znqi_read(206,2)=1065.0,qinit_read(206,2)=5.382e-04,
znqi_read(207,2)=1069.9,qinit_read(207,2)=5.412e-04,
znqi_read(208,2)=1074.7,qinit_read(208,2)=5.443e-04,
znqi_read(209,2)=1079.6,qinit_read(209,2)=5.474e-04,
znqi_read(210,2)=1084.6,qinit_read(210,2)=5.505e-04,
znqi_read(211,2)=1089.5,qinit_read(211,2)=5.537e-04,
znqi_read(212,2)=1094.6,qinit_read(212,2)=5.569e-04,
znqi_read(213,2)=1099.7,qinit_read(213,2)=5.601e-04,
znqi_read(214,2)=1104.9,qinit_read(214,2)=5.633e-04,
znqi_read(215,2)=1110.1,qinit_read(215,2)=5.666e-04,
znqi_read(216,2)=1115.4,qinit_read(216,2)=5.699e-04,
znqi_read(217,2)=1120.7,qinit_read(217,2)=5.732e-04,
znqi_read(218,2)=1126.0,qinit_read(218,2)=5.765e-04,
znqi_read(219,2)=1131.4,qinit_read(219,2)=5.798e-04,
znqi_read(220,2)=1136.7,qinit_read(220,2)=5.832e-04,
znqi_read(221,2)=1142.1,qinit_read(221,2)=5.865e-04,
znqi_read(222,2)=1147.4,qinit_read(222,2)=5.897e-04,
znqi_read(223,2)=1152.8,qinit_read(223,2)=5.930e-04,
znqi_read(224,2)=1158.1,qinit_read(224,2)=5.963e-04,
znqi_read(225,2)=1163.5,qinit_read(225,2)=0.000e+00,
znqi_read(226,2)=1168.9,qinit_read(226,2)=0.000e+00,
znqi_read(227,2)=1174.2,qinit_read(227,2)=0.000e+00,
znqi_read(228,2)=1179.6,qinit_read(228,2)=0.000e+00,
znqi_read(229,2)=1184.9,qinit_read(229,2)=0.000e+00,
znqi_read(230,2)=1190.2,qinit_read(230,2)=0.000e+00,
znqi_read(231,2)=1195.5,qinit_read(231,2)=0.000e+00,
znqi_read(232,2)=1200.8,qinit_read(232,2)=0.000e+00,
znqi_read(233,2)=1206.0,qinit_read(233,2)=0.000e+00,
znqi_read(234,2)=1211.3,qinit_read(234,2)=0.000e+00,
znqi_read(235,2)=1216.5,qinit_read(235,2)=0.000e+00,
znqi_read(236,2)=1221.8,qinit_read(236,2)=0.000e+00,
znqi_read(237,2)=1227.1,qinit_read(237,2)=0.000e+00,
znqi_read(238,2)=1232.4,qinit_read(238,2)=0.000e+00,
znqi_read(239,2)=1237.7,qinit_read(239,2)=0.000e+00,
znqi_read(240,2)=1243.0,qinit_read(240,2)=0.000e+00,
znqi_read(241,2)=1248.3,qinit_read(241,2)=0.000e+00,
znqi_read(242,2)=1253.6,qinit_read(242,2)=0.000e+00,
znqi_read(243,2)=1259.0,qinit_read(243,2)=0.000e+00,
znqi_read(244,2)=1264.3,qinit_read(244,2)=0.000e+00,
znqi_read(245,2)=1269.7,qinit_read(245,2)=0.000e+00,
znqi_read(246,2)=1275.2,qinit_read(246,2)=0.000e+00,
znqi_read(247,2)=1280.7,qinit_read(247,2)=0.000e+00,
znqi_read(248,2)=1286.1,qinit_read(248,2)=0.000e+00,
znqi_read(249,2)=1297.0,qinit_read(249,2)=0.000e+00,
znqi_read(250,2)=1302.5,qinit_read(250,2)=0.000e+00,
znqi_read(251,2)=1307.9,qinit_read(251,2)=0.000e+00,
znqi_read(252,2)=1313.2,qinit_read(252,2)=0.000e+00,
znqi_read(253,2)=1318.5,qinit_read(253,2)=0.000e+00,
znqi_read(254,2)=1323.9,qinit_read(254,2)=0.000e+00,
znqi_read(255,2)=1329.2,qinit_read(255,2)=0.000e+00,
znqi_read(256,2)=1334.5,qinit_read(256,2)=0.000e+00,
znqi_read(257,2)=1339.8,qinit_read(257,2)=0.000e+00,
znqi_read(258,2)=1345.0,qinit_read(258,2)=0.000e+00,
znqi_read(259,2)=1350.3,qinit_read(259,2)=0.000e+00,
znqi_read(260,2)=1355.5,qinit_read(260,2)=0.000e+00,
znqi_read(261,2)=1360.8,qinit_read(261,2)=0.000e+00,
znqi_read(262,2)=1366.0,qinit_read(262,2)=0.000e+00,
znqi_read(263,2)=1371.1,qinit_read(263,2)=0.000e+00,
znqi_read(264,2)=1376.3,qinit_read(264,2)=0.000e+00,
znqi_read(265,2)=1381.4,qinit_read(265,2)=0.000e+00,
znqi_read(266,2)=1386.6,qinit_read(266,2)=0.000e+00,
znqi_read(267,2)=1391.7,qinit_read(267,2)=0.000e+00,
znqi_read(268,2)=1396.8,qinit_read(268,2)=0.000e+00,
znqi_read(269,2)=1401.9,qinit_read(269,2)=0.000e+00,
znqi_read(270,2)=1407.0,qinit_read(270,2)=0.000e+00,
znqi_read(271,2)=1412.2,qinit_read(271,2)=0.000e+00,
znqi_read(272,2)=1417.4,qinit_read(272,2)=0.000e+00,
znqi_read(273,2)=1422.6,qinit_read(273,2)=0.000e+00,
znqi_read(274,2)=1427.9,qinit_read(274,2)=0.000e+00,
znqi_read(275,2)=1433.1,qinit_read(275,2)=0.000e+00,
znqi_read(276,2)=1438.3,qinit_read(276,2)=0.000e+00,
znqi_read(277,2)=1443.6,qinit_read(277,2)=0.000e+00,
znqi_read(278,2)=1448.8,qinit_read(278,2)=0.000e+00,
znqi_read(279,2)=1454.0,qinit_read(279,2)=0.000e+00,
znqi_read(280,2)=1459.3,qinit_read(280,2)=0.000e+00,
znqi_read(281,2)=1464.5,qinit_read(281,2)=0.000e+00,
znqi_read(282,2)=1469.7,qinit_read(282,2)=0.000e+00,
znqi_read(283,2)=1474.9,qinit_read(283,2)=0.000e+00,
znqi_read(284,2)=1480.0,qinit_read(284,2)=0.000e+00,
znqi_read(285,2)=1485.1,qinit_read(285,2)=0.000e+00,
znqi_read(286,2)=1490.2,qinit_read(286,2)=0.000e+00,
znqi_read(287,2)=1495.3,qinit_read(287,2)=0.000e+00,
znqi_read(288,2)=1500.4,qinit_read(288,2)=0.000e+00,
znqi_read(289,2)=1505.5,qinit_read(289,2)=0.000e+00,
znqi_read(290,2)=1510.7,qinit_read(290,2)=0.000e+00,
znqi_read(291,2)=1515.8,qinit_read(291,2)=0.000e+00,
znqi_read(292,2)=1520.9,qinit_read(292,2)=0.000e+00,
znqi_read(293,2)=1526.0,qinit_read(293,2)=0.000e+00,
znqi_read(294,2)=1531.1,qinit_read(294,2)=0.000e+00,
znqi_read(295,2)=1536.2,qinit_read(295,2)=0.000e+00,
znqi_read(296,2)=1541.3,qinit_read(296,2)=0.000e+00,
znqi_read(297,2)=1546.3,qinit_read(297,2)=0.000e+00,
znqi_read(298,2)=1551.3,qinit_read(298,2)=0.000e+00,
znqi_read(299,2)=1556.4,qinit_read(299,2)=0.000e+00,
znqi_read(300,2)=1561.4,qinit_read(300,2)=0.000e+00,
znqi_read(301,2)=1566.4,qinit_read(301,2)=0.000e+00,
znqi_read(302,2)=1571.4,qinit_read(302,2)=0.000e+00,
znqi_read(303,2)=1576.4,qinit_read(303,2)=0.000e+00,
znqi_read(304,2)=1581.5,qinit_read(304,2)=0.000e+00,
znqi_read(305,2)=1586.5,qinit_read(305,2)=0.000e+00,
znqi_read(306,2)=1591.6,qinit_read(306,2)=0.000e+00,
znqi_read(307,2)=1596.6,qinit_read(307,2)=0.000e+00,
znqi_read(308,2)=1601.6,qinit_read(308,2)=0.000e+00,
znqi_read(309,2)=1606.7,qinit_read(309,2)=0.000e+00,
znqi_read(310,2)=1611.7,qinit_read(310,2)=0.000e+00,
znqi_read(311,2)=1616.7,qinit_read(311,2)=0.000e+00,
znqi_read(312,2)=1621.7,qinit_read(312,2)=0.000e+00,
znqi_read(313,2)=1626.7,qinit_read(313,2)=0.000e+00,
znqi_read(314,2)=1631.7,qinit_read(314,2)=0.000e+00,
znqi_read(315,2)=1636.8,qinit_read(315,2)=0.000e+00,
znqi_read(316,2)=1641.9,qinit_read(316,2)=0.000e+00,
znqi_read(317,2)=1647.0,qinit_read(317,2)=0.000e+00,
znqi_read(318,2)=1652.2,qinit_read(318,2)=0.000e+00,
znqi_read(319,2)=1657.3,qinit_read(319,2)=0.000e+00,
znqi_read(320,2)=1662.4,qinit_read(320,2)=0.000e+00,
znqi_read(321,2)=1667.4,qinit_read(321,2)=0.000e+00,
znqi_read(322,2)=1672.5,qinit_read(322,2)=0.000e+00,
znqi_read(323,2)=1677.5,qinit_read(323,2)=0.000e+00,
znqi_read(324,2)=1682.5,qinit_read(324,2)=0.000e+00,
znqi_read(325,2)=1687.5,qinit_read(325,2)=0.000e+00,
znqi_read(326,2)=1692.5,qinit_read(326,2)=0.000e+00,
znqi_read(327,2)=1697.6,qinit_read(327,2)=0.000e+00,
znqi_read(328,2)=1702.6,qinit_read(328,2)=0.000e+00,
znqi_read(329,2)=1707.6,qinit_read(329,2)=0.000e+00,
znqi_read(330,2)=1712.7,qinit_read(330,2)=0.000e+00,
znqi_read(331,2)=1717.7,qinit_read(331,2)=0.000e+00,
znqi_read(332,2)=1722.8,qinit_read(332,2)=0.000e+00,
znqi_read(333,2)=1727.9,qinit_read(333,2)=0.000e+00,
znqi_read(334,2)=1733.0,qinit_read(334,2)=0.000e+00,
znqi_read(335,2)=1738.2,qinit_read(335,2)=0.000e+00,
znqi_read(336,2)=1743.3,qinit_read(336,2)=0.000e+00,
znqi_read(337,2)=1748.5,qinit_read(337,2)=0.000e+00,
znqi_read(338,2)=1753.6,qinit_read(338,2)=0.000e+00,
znqi_read(339,2)=1758.8,qinit_read(339,2)=0.000e+00,
znqi_read(340,2)=1763.9,qinit_read(340,2)=0.000e+00,
znqi_read(341,2)=1769.1,qinit_read(341,2)=0.000e+00,
znqi_read(342,2)=1774.2,qinit_read(342,2)=0.000e+00,
znqi_read(343,2)=1779.3,qinit_read(343,2)=0.000e+00,
znqi_read(344,2)=1784.4,qinit_read(344,2)=0.000e+00,
znqi_read(345,2)=1789.4,qinit_read(345,2)=0.000e+00,
znqi_read(346,2)=1794.5,qinit_read(346,2)=0.000e+00,
znqi_read(347,2)=1799.5,qinit_read(347,2)=0.000e+00,
znqi_read(348,2)=1804.5,qinit_read(348,2)=0.000e+00,
znqi_read(349,2)=1809.6,qinit_read(349,2)=0.000e+00,
znqi_read(350,2)=1814.6,qinit_read(350,2)=0.000e+00,
znqi_read(351,2)=1819.6,qinit_read(351,2)=0.000e+00,
znqi_read(352,2)=1824.7,qinit_read(352,2)=0.000e+00,
znqi_read(353,2)=1829.9,qinit_read(353,2)=0.000e+00,
znqi_read(354,2)=1835.0,qinit_read(354,2)=0.000e+00,
znqi_read(355,2)=1840.2,qinit_read(355,2)=0.000e+00,
znqi_read(356,2)=1845.4,qinit_read(356,2)=0.000e+00,
znqi_read(357,2)=1850.6,qinit_read(357,2)=0.000e+00,
znqi_read(358,2)=1855.8,qinit_read(358,2)=0.000e+00,
znqi_read(359,2)=1861.0,qinit_read(359,2)=0.000e+00,
znqi_read(360,2)=1866.2,qinit_read(360,2)=0.000e+00,
znqi_read(361,2)=1871.3,qinit_read(361,2)=0.000e+00,
znqi_read(362,2)=1876.5,qinit_read(362,2)=0.000e+00,
znqi_read(363,2)=1881.6,qinit_read(363,2)=0.000e+00,
znqi_read(364,2)=1886.8,qinit_read(364,2)=0.000e+00,
znqi_read(365,2)=1891.9,qinit_read(365,2)=0.000e+00,
znqi_read(366,2)=1897.1,qinit_read(366,2)=0.000e+00,
znqi_read(367,2)=1902.2,qinit_read(367,2)=0.000e+00,
znqi_read(368,2)=1907.4,qinit_read(368,2)=0.000e+00,
znqi_read(369,2)=1912.5,qinit_read(369,2)=0.000e+00,
znqi_read(370,2)=1917.7,qinit_read(370,2)=0.000e+00,
znqi_read(371,2)=1922.9,qinit_read(371,2)=0.000e+00,
znqi_read(372,2)=1928.1,qinit_read(372,2)=0.000e+00,
znqi_read(373,2)=1933.4,qinit_read(373,2)=0.000e+00,
znqi_read(374,2)=1938.6,qinit_read(374,2)=0.000e+00,
znqi_read(375,2)=1943.9,qinit_read(375,2)=0.000e+00,
znqi_read(376,2)=1949.2,qinit_read(376,2)=0.000e+00,
znqi_read(377,2)=1954.4,qinit_read(377,2)=0.000e+00,
znqi_read(378,2)=1959.7,qinit_read(378,2)=0.000e+00,
znqi_read(379,2)=1964.9,qinit_read(379,2)=0.000e+00,
znqi_read(380,2)=1970.1,qinit_read(380,2)=0.000e+00,
znqi_read(381,2)=1975.3,qinit_read(381,2)=0.000e+00,
znqi_read(382,2)=1980.5,qinit_read(382,2)=0.000e+00,
znqi_read(383,2)=1985.7,qinit_read(383,2)=0.000e+00,
znqi_read(384,2)=1990.9,qinit_read(384,2)=0.000e+00,
znqi_read(385,2)=1996.1,qinit_read(385,2)=0.000e+00,
znqi_read(386,2)=2001.3,qinit_read(386,2)=0.000e+00,
znqi_read(387,2)=2006.5,qinit_read(387,2)=0.000e+00,
znqi_read(388,2)=2011.8,qinit_read(388,2)=0.000e+00,
znqi_read(389,2)=2017.1,qinit_read(389,2)=0.000e+00,
znqi_read(390,2)=2022.5,qinit_read(390,2)=0.000e+00,
znqi_read(391,2)=2027.9,qinit_read(391,2)=0.000e+00,
znqi_read(392,2)=2033.2,qinit_read(392,2)=0.000e+00,
znqi_read(393,2)=2038.6,qinit_read(393,2)=0.000e+00,
znqi_read(394,2)=2043.9,qinit_read(394,2)=0.000e+00,
znqi_read(395,2)=2049.1,qinit_read(395,2)=0.000e+00,
znqi_read(396,2)=2054.2,qinit_read(396,2)=0.000e+00,
znqi_read(397,2)=2059.3,qinit_read(397,2)=0.000e+00,
znqi_read(398,2)=2064.4,qinit_read(398,2)=0.000e+00,
znqi_read(399,2)=2069.5,qinit_read(399,2)=0.000e+00,
znqi_read(400,2)=2074.6,qinit_read(400,2)=0.000e+00,
znqi_read(401,2)=2079.8,qinit_read(401,2)=0.000e+00,
znqi_read(402,2)=2085.1,qinit_read(402,2)=0.000e+00,
znqi_read(403,2)=2090.4,qinit_read(403,2)=0.000e+00,
znqi_read(404,2)=2095.7,qinit_read(404,2)=0.000e+00,
znqi_read(405,2)=2101.0,qinit_read(405,2)=0.000e+00,
znqi_read(406,2)=2106.4,qinit_read(406,2)=0.000e+00,
znqi_read(407,2)=2111.7,qinit_read(407,2)=0.000e+00,
znqi_read(408,2)=2117.0,qinit_read(408,2)=0.000e+00,
znqi_read(409,2)=2122.3,qinit_read(409,2)=0.000e+00,
znqi_read(410,2)=2127.6,qinit_read(410,2)=0.000e+00,
znqi_read(411,2)=2132.9,qinit_read(411,2)=0.000e+00,
znqi_read(412,2)=2138.1,qinit_read(412,2)=0.000e+00,
znqi_read(413,2)=2143.3,qinit_read(413,2)=0.000e+00,
znqi_read(414,2)=2148.5,qinit_read(414,2)=0.000e+00,
znqi_read(415,2)=2153.7,qinit_read(415,2)=0.000e+00,
znqi_read(416,2)=2158.9,qinit_read(416,2)=0.000e+00,
znqi_read(417,2)=2164.0,qinit_read(417,2)=0.000e+00,
znqi_read(418,2)=2169.2,qinit_read(418,2)=0.000e+00,
znqi_read(419,2)=2174.4,qinit_read(419,2)=0.000e+00,
znqi_read(420,2)=2179.6,qinit_read(420,2)=0.000e+00,
znqi_read(421,2)=2184.7,qinit_read(421,2)=0.000e+00,
znqi_read(422,2)=2189.8,qinit_read(422,2)=0.000e+00,
znqi_read(423,2)=2194.9,qinit_read(423,2)=0.000e+00,
znqi_read(424,2)=2200.0,qinit_read(424,2)=0.000e+00,
znqi_read(425,2)=2205.2,qinit_read(425,2)=0.000e+00,
znqi_read(426,2)=2210.4,qinit_read(426,2)=0.000e+00,
znqi_read(427,2)=2215.7,qinit_read(427,2)=0.000e+00,
znqi_read(428,2)=2221.0,qinit_read(428,2)=0.000e+00,
znqi_read(429,2)=2226.3,qinit_read(429,2)=0.000e+00,
znqi_read(430,2)=2231.6,qinit_read(430,2)=0.000e+00,
znqi_read(431,2)=2237.0,qinit_read(431,2)=0.000e+00,
znqi_read(432,2)=2242.3,qinit_read(432,2)=0.000e+00,
znqi_read(433,2)=2247.6,qinit_read(433,2)=0.000e+00,
znqi_read(434,2)=2252.8,qinit_read(434,2)=0.000e+00,
znqi_read(435,2)=2258.1,qinit_read(435,2)=0.000e+00,
znqi_read(436,2)=2263.3,qinit_read(436,2)=0.000e+00,
znqi_read(437,2)=2268.5,qinit_read(437,2)=0.000e+00,
znqi_read(438,2)=2273.7,qinit_read(438,2)=0.000e+00,
znqi_read(439,2)=2278.9,qinit_read(439,2)=0.000e+00,
znqi_read(440,2)=2284.1,qinit_read(440,2)=0.000e+00,
znqi_read(441,2)=2289.3,qinit_read(441,2)=0.000e+00,
znqi_read(442,2)=2294.6,qinit_read(442,2)=0.000e+00,
znqi_read(443,2)=2299.8,qinit_read(443,2)=0.000e+00,
znqi_read(444,2)=2305.1,qinit_read(444,2)=0.000e+00,
znqi_read(445,2)=2310.3,qinit_read(445,2)=0.000e+00,
znqi_read(446,2)=2315.6,qinit_read(446,2)=0.000e+00,
znqi_read(447,2)=2320.8,qinit_read(447,2)=0.000e+00,
znqi_read(448,2)=2326.1,qinit_read(448,2)=0.000e+00,
znqi_read(449,2)=2331.4,qinit_read(449,2)=0.000e+00,
znqi_read(450,2)=2336.7,qinit_read(450,2)=0.000e+00,
znqi_read(451,2)=2342.0,qinit_read(451,2)=0.000e+00,
znqi_read(452,2)=2347.2,qinit_read(452,2)=0.000e+00,
znqi_read(453,2)=2352.5,qinit_read(453,2)=0.000e+00,
znqi_read(454,2)=2357.7,qinit_read(454,2)=0.000e+00,
znqi_read(455,2)=2363.0,qinit_read(455,2)=0.000e+00,
znqi_read(456,2)=2368.2,qinit_read(456,2)=0.000e+00,
znqi_read(457,2)=2373.5,qinit_read(457,2)=0.000e+00,
znqi_read(458,2)=2378.8,qinit_read(458,2)=0.000e+00,
znqi_read(459,2)=2384.1,qinit_read(459,2)=0.000e+00,
znqi_read(460,2)=2389.4,qinit_read(460,2)=0.000e+00,
znqi_read(461,2)=2394.7,qinit_read(461,2)=0.000e+00,
znqi_read(462,2)=2400.0,qinit_read(462,2)=0.000e+00,
znqi_read(463,2)=2405.3,qinit_read(463,2)=0.000e+00,
znqi_read(464,2)=2410.6,qinit_read(464,2)=0.000e+00,
znqi_read(465,2)=2415.9,qinit_read(465,2)=0.000e+00,
znqi_read(466,2)=2421.2,qinit_read(466,2)=0.000e+00,
znqi_read(467,2)=2426.4,qinit_read(467,2)=0.000e+00,
znqi_read(468,2)=2431.7,qinit_read(468,2)=0.000e+00,
znqi_read(469,2)=2437.1,qinit_read(469,2)=0.000e+00,
znqi_read(470,2)=2442.4,qinit_read(470,2)=0.000e+00,
znqi_read(471,2)=2447.8,qinit_read(471,2)=0.000e+00,
znqi_read(472,2)=2453.1,qinit_read(472,2)=0.000e+00,
znqi_read(473,2)=2458.5,qinit_read(473,2)=0.000e+00,
znqi_read(474,2)=2463.9,qinit_read(474,2)=0.000e+00,
znqi_read(475,2)=2469.3,qinit_read(475,2)=0.000e+00,
znqi_read(476,2)=2474.6,qinit_read(476,2)=0.000e+00,
znqi_read(477,2)=2480.0,qinit_read(477,2)=0.000e+00,
znqi_read(478,2)=2485.4,qinit_read(478,2)=0.000e+00,
znqi_read(479,2)=2490.8,qinit_read(479,2)=0.000e+00,
znqi_read(480,2)=2496.1,qinit_read(480,2)=0.000e+00,
znqi_read(481,2)=2501.5,qinit_read(481,2)=0.000e+00,
znqi_read(482,2)=2506.8,qinit_read(482,2)=0.000e+00,
znqi_read(483,2)=2512.1,qinit_read(483,2)=0.000e+00,
znqi_read(484,2)=2517.5,qinit_read(484,2)=0.000e+00,
znqi_read(485,2)=2522.8,qinit_read(485,2)=0.000e+00,
znqi_read(486,2)=2528.2,qinit_read(486,2)=0.000e+00,
znqi_read(487,2)=2533.5,qinit_read(487,2)=0.000e+00,
znqi_read(488,2)=2538.9,qinit_read(488,2)=0.000e+00,
znqi_read(489,2)=2544.2,qinit_read(489,2)=0.000e+00,
znqi_read(490,2)=2549.5,qinit_read(490,2)=0.000e+00,
znqi_read(491,2)=2554.9,qinit_read(491,2)=0.000e+00,
znqi_read(492,2)=2560.2,qinit_read(492,2)=0.000e+00,
znqi_read(493,2)=2565.6,qinit_read(493,2)=0.000e+00,
znqi_read(494,2)=2570.9,qinit_read(494,2)=0.000e+00,
znqi_read(495,2)=2576.3,qinit_read(495,2)=0.000e+00,
znqi_read(496,2)=2581.6,qinit_read(496,2)=0.000e+00,
znqi_read(497,2)=2587.0,qinit_read(497,2)=0.000e+00,
znqi_read(498,2)=2592.4,qinit_read(498,2)=0.000e+00,
znqi_read(499,2)=2597.7,qinit_read(499,2)=0.000e+00,
znqi_read(500,2)=2603.2,qinit_read(500,2)=0.000e+00,
znqi_read(501,2)=2608.6,qinit_read(501,2)=0.000e+00,
znqi_read(502,2)=2614.0,qinit_read(502,2)=0.000e+00,
znqi_read(503,2)=2619.4,qinit_read(503,2)=0.000e+00,
znqi_read(504,2)=2624.8,qinit_read(504,2)=0.000e+00,
znqi_read(505,2)=2630.2,qinit_read(505,2)=0.000e+00,
znqi_read(506,2)=2635.5,qinit_read(506,2)=0.000e+00,
znqi_read(507,2)=2640.9,qinit_read(507,2)=0.000e+00,
znqi_read(508,2)=2646.2,qinit_read(508,2)=0.000e+00,
znqi_read(509,2)=2651.5,qinit_read(509,2)=0.000e+00,
znqi_read(510,2)=2656.8,qinit_read(510,2)=0.000e+00,
znqi_read(511,2)=2662.1,qinit_read(511,2)=0.000e+00,
znqi_read(512,2)=2667.4,qinit_read(512,2)=0.000e+00,
znqi_read(513,2)=2672.7,qinit_read(513,2)=0.000e+00,
znqi_read(514,2)=2678.1,qinit_read(514,2)=0.000e+00,
znqi_read(515,2)=2683.5,qinit_read(515,2)=0.000e+00,
znqi_read(516,2)=2688.9,qinit_read(516,2)=0.000e+00,
znqi_read(517,2)=2694.4,qinit_read(517,2)=0.000e+00,
znqi_read(518,2)=2699.8,qinit_read(518,2)=0.000e+00,
znqi_read(519,2)=2705.2,qinit_read(519,2)=0.000e+00,
znqi_read(520,2)=2710.6,qinit_read(520,2)=0.000e+00,
znqi_read(521,2)=2716.0,qinit_read(521,2)=0.000e+00,
znqi_read(522,2)=2721.3,qinit_read(522,2)=0.000e+00,
znqi_read(523,2)=2726.7,qinit_read(523,2)=0.000e+00,
znqi_read(524,2)=2732.1,qinit_read(524,2)=0.000e+00,
znqi_read(525,2)=2737.5,qinit_read(525,2)=0.000e+00,
znqi_read(526,2)=2742.9,qinit_read(526,2)=0.000e+00,
znqi_read(527,2)=2748.3,qinit_read(527,2)=0.000e+00,
znqi_read(528,2)=2753.7,qinit_read(528,2)=0.000e+00,
znqi_read(529,2)=2759.2,qinit_read(529,2)=0.000e+00,
znqi_read(530,2)=2764.6,qinit_read(530,2)=0.000e+00,
znqi_read(531,2)=2770.1,qinit_read(531,2)=0.000e+00,
znqi_read(532,2)=2775.5,qinit_read(532,2)=0.000e+00,
znqi_read(533,2)=2781.0,qinit_read(533,2)=0.000e+00,
znqi_read(534,2)=2786.4,qinit_read(534,2)=0.000e+00,
znqi_read(535,2)=2791.8,qinit_read(535,2)=0.000e+00,
znqi_read(536,2)=2797.1,qinit_read(536,2)=0.000e+00,
znqi_read(537,2)=2802.5,qinit_read(537,2)=0.000e+00,
znqi_read(538,2)=2808.0,qinit_read(538,2)=0.000e+00,
znqi_read(539,2)=2813.4,qinit_read(539,2)=0.000e+00,
znqi_read(540,2)=2818.9,qinit_read(540,2)=0.000e+00,
znqi_read(541,2)=2824.4,qinit_read(541,2)=0.000e+00,
znqi_read(542,2)=2840.8,qinit_read(542,2)=0.000e+00,
znqi_read(543,2)=2846.2,qinit_read(543,2)=0.000e+00,
znqi_read(544,2)=2851.7,qinit_read(544,2)=0.000e+00,
znqi_read(545,2)=2857.1,qinit_read(545,2)=0.000e+00,
znqi_read(546,2)=2862.5,qinit_read(546,2)=0.000e+00,
znqi_read(547,2)=2867.9,qinit_read(547,2)=0.000e+00,
znqi_read(548,2)=2873.3,qinit_read(548,2)=0.000e+00,
znqi_read(549,2)=2878.7,qinit_read(549,2)=0.000e+00,
znqi_read(550,2)=2884.2,qinit_read(550,2)=0.000e+00,
znqi_read(551,2)=2889.6,qinit_read(551,2)=0.000e+00,
znqi_read(552,2)=2895.1,qinit_read(552,2)=0.000e+00,
znqi_read(553,2)=2900.5,qinit_read(553,2)=0.000e+00,
znqi_read(554,2)=2906.0,qinit_read(554,2)=0.000e+00,
znqi_read(555,2)=2911.5,qinit_read(555,2)=0.000e+00,
znqi_read(556,2)=2917.0,qinit_read(556,2)=0.000e+00,
znqi_read(557,2)=2922.5,qinit_read(557,2)=0.000e+00,
znqi_read(558,2)=2928.0,qinit_read(558,2)=0.000e+00,
znqi_read(559,2)=2933.5,qinit_read(559,2)=0.000e+00,
znqi_read(560,2)=2939.0,qinit_read(560,2)=0.000e+00,
znqi_read(561,2)=2944.5,qinit_read(561,2)=0.000e+00,
znqi_read(562,2)=2950.1,qinit_read(562,2)=0.000e+00,
znqi_read(563,2)=2955.6,qinit_read(563,2)=0.000e+00,
znqi_read(564,2)=2961.2,qinit_read(564,2)=0.000e+00,
znqi_read(565,2)=2966.7,qinit_read(565,2)=0.000e+00,
znqi_read(566,2)=2972.2,qinit_read(566,2)=0.000e+00,
znqi_read(567,2)=2977.8,qinit_read(567,2)=0.000e+00,
znqi_read(568,2)=2983.3,qinit_read(568,2)=0.000e+00,
znqi_read(569,2)=2988.7,qinit_read(569,2)=0.000e+00,
znqi_read(570,2)=2994.2,qinit_read(570,2)=0.000e+00,
znqi_read(571,2)=2999.6,qinit_read(571,2)=0.000e+00,
znqi_read(572,2)=3005.0,qinit_read(572,2)=0.000e+00,


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
