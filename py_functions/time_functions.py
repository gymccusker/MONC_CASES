"""
Functions to convert timestamps
==============================

"""

from __future__ import print_function
import time
import datetime
import numpy as np
import pandas as pd

def calcTime_Mat2DOY(matlab_time):

        #### EXAMPLE OF USE:
        #### pytime = calcTime_Mat2DOY(matlab_time)

    print ('Converting MATLAB timesteps to DOY:')

    timestamps = pd.to_datetime(matlab_time-719529, unit='D')
    python_time = timestamps.dayofyear + (timestamps.hour / 24.0) + (timestamps.minute / 1440.0) + (timestamps.second / 86400.0)

    return python_time

def calcTime_Date2DOY(date):

    ####        date should be forematted as YYYYmmDD

    print ('Converting date to DOY:')

    mm = date[4:6]      #### month
    DD = date[6:8]      #### day
    refDateAug = 226       #### Aug reference date for drift: 14th Aug 2018
    refDateSep = 243       #### Sep reference date for drift: 1st Sep 2018

    if mm == '08':
        doy = (float(DD) - 14.0) + refDateAug
    elif mm == '09':
        doy = float(DD) + refDateSep
    else:
        print ('****Date not valid with this function****')

    print ('----')
    print ('Date = ', date)
    print ('DOY = ', doy)
    print ('')

    return doy

def serial_date_to_string(srl_no):

    '''
    Function to convert serial date (wrt 1981-10-01) to string
    '''

    new_date = datetime.datetime(1981,10,1,0,0) + datetime.timedelta(srl_no)
    return new_date.strftime("%Y-%m-%d")

def serial_date_to_doy(srl_no):

    '''
    Function to convert serial date (wrt 1981-10-01) to doy (in 2018)
    '''

    new_date = datetime.datetime(1981,10,1,0,0) + datetime.timedelta(srl_no)
    doy_temp = new_date - datetime.datetime(2018,1,1,0,0)
    doy = doy_temp.days + 1

    return doy
