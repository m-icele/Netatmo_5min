# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    99_same_p99_appearance.py
Purpose: check whether p99 values occur regionally at the same time
Created on: 14.05.2021
"""

__author__ = 'Micha Eisele'
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = 'micha.eisele@iws.uni-stuttgart.de'
__version__ = 0.1
__last_update__ = '14.05.2021'

# ==============================================================================
# import
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys
import tables
import scipy.stats as sst
from scipy import interpolate as sci
import scipy.spatial as ssp
import fiona

sys.path.append(r'D:\py-sele')
import mail

# ==============================================================================
# functions
# ==============================================================================
def build_edf(measurements, vInv=False):
    npF1 = np.array(measurements)
    nPts = np.shape(measurements)[0]
    ErwTr = 1. / (2. * nPts)
    EDFWert = 1. / (nPts)
    asort = np.arange((ErwTr), (1.), (EDFWert))
    ind = sst.rankdata(npF1, method='ordinal')

    StdF1 = asort[np.int_(ind) - 1]

    if vInv is True:
        StdF1 = 1 - StdF1
    return StdF1

# ==============================================================================
# settings
# ==============================================================================
# path to dwd
path_dwd = r"D:\bwsyncandshare\Netatmo_DWD\03_dwd\DWD_5min_to_1hour.h5"

# path to echaz
path_catchment = (r"D:\2020_DFG_Netatmo\03_data\00_other\00_shapefiles\DEU_adm"
                  r"\DEU_adm0.shp")

savepath = r'D:\Netatmo_5min\03_results\99_same_p99_appearance'

year = 2019

# ==============================================================================
# process
# ==============================================================================
dwd = tables.open_file(path_dwd)

# open shape
ishape = fiona.open(path_catchment)
first = ishape.next()

# set up dwd tree
dwd_coords = np.vstack((dwd.root.coord.lat[:],
                        dwd.root.coord.lon[:])).T

p99_value = np.zeros(dwd.root.name.shape[0])
p99_indi = np.zeros((dwd.root.name.shape[0], 5136))

for dwd_id, dwd_name in enumerate(dwd.root.name.read()):
    print(dwd_id)
    df_dwd = pd.DataFrame(index=dwd.root.timestamps.isoformat.read(),
                          columns=['dwd'],
                          data=dwd.root.data[:, dwd_id])

    df_dwd.index = pd.to_datetime(
        df_dwd.index.str.decode('utf-8'))

    # select year
    df_dwd = df_dwd[
             '{}-04-01 01:00:00'.format(year):'{}-11-01 00:00:00'.format(
                 year)]

    value_mask = df_dwd.isna().values[:, 0]

    dwd_data = df_dwd[~df_dwd.isna()].values[:, 0]

    # build edf
    try:
        dwd_edf = build_edf(np.sort(dwd_data))
    except ZeroDivisionError:
        if value_mask.sum() == 0:
            continue

    # get 0.99
    try:
        dwd_tresh = sci.interp1d(dwd_edf, np.sort(dwd_data))(0.99)
        p99_value[dwd_id] = dwd_tresh
    except:
        continue

    # set indicator timeseries
    dwd_indi_p99 = np.zeros_like(dwd_data)
    dwd_indi_p99[dwd_data >= dwd_tresh] = 1

    p99_indi[dwd_id, :] = dwd_indi_p99

    plt.scatter(dwd_coords[dwd_id, 1],
                dwd_coords[dwd_id, 0],
                c=p99_value[dwd_id],
                vmin=0,
                vmax=10)

plt.colorbar()
plt.show()
print()

p99_indi_sum = np.sum(p99_indi, axis=0)
p99_idx = np.where(p99_indi_sum > 0)[0]

for idx in p99_idx:
    plt.scatter(dwd_coords[:, 1],
                dwd_coords[:, 0],
                c=p99_indi[:, idx],
                vmin=0,
                vmax=1,
                s=2,
                cmap='binary')

    for n, i_poly in enumerate(first['geometry']['coordinates']):
        # print(n)
        plt.plot(np.array(i_poly)[0, :, 0],
                 np.array(i_poly)[0, :, 1],
                 linestyle='-',
                 linewidth=0.8,
                 color='black',
                 )
    plt.title(df_dwd.iloc[idx].name.isoformat())

    plt.axis('equal')

    plt.tight_layout()

    plt.savefig(os.path.join(savepath,
                             '{}.png'.format(idx)),
                dpi=300)
    plt.close()




mail.status_mail(
    subject='99_same_p99_appearance.py finished %s' % datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'))
