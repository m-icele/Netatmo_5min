# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    01_misplaced_filter.py
Purpose: PWS are often misplaced or shifted on the map resulting in the data
         that does not match the location. This filter identifies such time
         series based on indicator correlations with other PWSs and potentially
         allows for a correct spatial allocation of the data.
Created on: 12.05.2021
"""

__author__ = 'Micha Eisele'
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = 'micha.eisele@iws.uni-stuttgart.de'
__version__ = 0.1
__last_update__ = '12.05.2021'

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
import scipy.spatial as ssp
import scipy.stats as sst
from scipy import interpolate as sci
import multiprocessing as mp

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


def calc_p0_for_given_coordinates(args):
    (dwd_tree, net_north, net_east, savepath, path_dwd, df_netatmo,
     stn_id) = args

    print(stn_id)

    dwd = tables.open_file(path_dwd)

    nearest_dwds = dwd_tree.query(np.hstack((net_north, net_east)),
                                  k=100)[1]

    selected_dwds = np.hstack((nearest_dwds[:5], nearest_dwds[5::5]))

    # initialize lists
    distance_list = []
    ik_list = []

    for dwd_id in selected_dwds:
        # calc distance
        dwd_north = dwd.root.coord.northing[dwd_id]
        dwd_east = dwd.root.coord.easting[dwd_id]

        distance = np.sqrt(
            (net_east - dwd_east) ** 2 + (net_north - dwd_north) ** 2)

        df_dwd = pd.DataFrame(index=dwd.root.timestamps.isoformat.read(),
                              columns=['dwd'],
                              data=dwd.root.data[:, dwd_id])

        df_dwd.index = pd.to_datetime(
            df_dwd.index.str.decode('utf-8'))

        df_combine = pd.concat([df_dwd, df_netatmo], join='inner',
                               axis=1)

        both_value_mask = (np.isnan(df_combine).sum(axis=1) == 0).values

        if both_value_mask.sum() < 5:
            continue

        dwd_data = df_combine[both_value_mask].values[:, 0]
        net_data = df_combine[both_value_mask].values[:, 1]

        # set indicator timeseries p0
        dwd_indi = np.zeros_like(dwd_data)
        dwd_indi[dwd_data == 0] = 1

        net_indi = np.zeros_like(net_data)
        net_indi[net_data == 0] = 1

        plt.scatter(distance,
                    np.corrcoef(dwd_indi, net_indi)[0, 1],
                    marker='o', c='black', s=2)

        distance_list.append(distance)
        ik_list.append(np.corrcoef(dwd_indi, net_indi)[0, 1])

    idx = np.argsort(distance_list)
    distance = np.array(distance_list)[idx]
    ik_arr = np.array(ik_list)[idx]

    try:
        if np.where(ik_arr == np.max(ik_arr))[0][0] < 3:
            plt.close()
            dwd.close()
            return
    except (ValueError, IndexError):
        dwd.close()
        return

    # initialize lists
    distance_list = []
    ik_list = []

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)

    # calc p0 for all dwd stations
    for dwd_id in range(dwd.root.name.shape[0]):
        # calc distance
        dwd_north = dwd.root.coord.northing[dwd_id]
        dwd_east = dwd.root.coord.easting[dwd_id]

        distance = np.sqrt(
            (net_east - dwd_east) ** 2 + (net_north - dwd_north) ** 2)

        df_dwd = pd.DataFrame(
            index=dwd.root.timestamps.isoformat.read(),
            columns=['dwd'],
            data=dwd.root.data[:, dwd_id])

        df_dwd.index = pd.to_datetime(
            df_dwd.index.str.decode('utf-8'))

        df_combine = pd.concat([df_dwd, df_netatmo], join='inner',
                               axis=1)

        both_value_mask = (np.isnan(df_combine).sum(axis=1) == 0).values

        if both_value_mask.sum() < 5:
            continue

        dwd_data = df_combine[both_value_mask].values[:, 0]
        net_data = df_combine[both_value_mask].values[:, 1]

        # set indicator timeseries p0
        dwd_indi = np.zeros_like(dwd_data)
        dwd_indi[dwd_data == 0] = 1

        net_indi = np.zeros_like(net_data)
        net_indi[net_data == 0] = 1

        ax[0].scatter(distance,
                    np.corrcoef(dwd_indi, net_indi)[0, 1],
                    marker='o', c='black', s=2)

        distance_list.append(distance)
        ik_list.append(np.corrcoef(dwd_indi, net_indi)[0, 1])

        # build edf
        try:
            dwd_edf = build_edf(np.sort(dwd_data))
            net_edf = build_edf(np.sort(net_data))
        except ZeroDivisionError:
            if both_value_mask.sum() == 0:
                # print(dwd_id, 'No time overlap')
                continue

        # get 0.99
        try:
            dwd_tresh = sci.interp1d(dwd_edf, np.sort(dwd_data))(0.99)
            dwd_tresh1 = sci.interp1d(net_edf, np.sort(net_data))(0.99)
        except:
            continue

        # set indicator timeseries
        dwd_indi_p99 = np.zeros_like(dwd_data)
        dwd_indi_p99[dwd_data >= dwd_tresh] = 1

        net_indi_p99 = np.zeros_like(net_data)
        net_indi_p99[net_data >= dwd_tresh1] = 1

        ax[1].scatter(distance,
                    np.corrcoef(dwd_indi_p99, net_indi_p99)[0, 1],
                    marker='o', c='black', s=2)

    plt.savefig(os.path.join(savepath,
                             'ind_corr_p0_dwd_{}.png'.format(stn_id)))
    plt.close()

    dwd.close()
    return


# ==============================================================================
# settings
# ==============================================================================
# path to dwd
path_dwd = r"D:\bwsyncandshare\Netatmo_DWD\03_dwd\DWD_5min_to_1hour.h5"

# path to netatmo data
path_netatmo = (r"D:\bwsyncandshare\Netatmo_DWD\01_netatmo"
                r"\netatmo_Germany_5min_to_1hour_filter_00.h5")

savepath = (r'D:\Netatmo_5min\03_results\01_misplaced_filter')

x_percent_dwd = 0.1


# ==============================================================================
# process
# ==============================================================================
def process_manager():
    # open data sets
    netatmo_hf = tables.open_file(path_netatmo)
    dwd = tables.open_file(path_dwd)

    # set up dwd tree
    dwd_coords = np.vstack((dwd.root.coord.northing[:],
                            dwd.root.coord.easting[:])).T

    dwd_tree = ssp.cKDTree(dwd_coords)

    # initialize multiprocessing
    my_pool = mp.Pool(3)
    args = ()

    # 1. identify misplaced stations
    for stn_id, stn_name in enumerate(netatmo_hf.root.name.read()):
        if stn_id < 40:
            continue

        if stn_id > 200:
            continue

        print(stn_id, stn_name)

        # get data for current station
        df_netatmo = pd.DataFrame(
            index=netatmo_hf.root.timestamps.isoformat.read(),
            columns=['netatmo'],
            data=netatmo_hf.root.data[:, stn_id])

        df_netatmo.index = pd.to_datetime(df_netatmo.index.str.decode('utf-8'))

        # get coordinates
        net_north = netatmo_hf.root.coord.northing[stn_id]
        net_east = netatmo_hf.root.coord.easting[stn_id]

        args += (
            (dwd_tree, net_north, net_east, savepath, path_dwd, df_netatmo,
             stn_id),)

    my_pool.map(calc_p0_for_given_coordinates, args)

    my_pool.close()
    my_pool.join()


if __name__ == '__main__':
    start = datetime.datetime.now()
    process_manager()

    print(datetime.datetime.now() - start)

    # nearest_dwds = dwd_tree.query(np.hstack((net_north, net_east)),
    #                               k=100)[1]
    #
    # selected_dwds = np.hstack((nearest_dwds[:5], nearest_dwds[5::5]))
    #
    # for dwd_id in selected_dwds:
    #     # calc distance
    #     dwd_north = dwd.root.coord.northing[dwd_id]
    #     dwd_east = dwd.root.coord.easting[dwd_id]
    #
    #     distance = np.sqrt(
    #         (net_east - dwd_east) ** 2 + (net_north - dwd_north) ** 2)
    #
    #     df_dwd = pd.DataFrame(index=dwd.root.timestamps.isoformat.read(),
    #                           columns=['dwd'],
    #                           data=dwd.root.data[:, dwd_id])
    #
    #     df_dwd.index = pd.to_datetime(
    #         df_dwd.index.str.decode('utf-8'))
    #
    #     df_combine = pd.concat([df_dwd, df_netatmo], join='inner',
    #                            axis=1)
    #
    #     both_value_mask = (np.isnan(df_combine).sum(axis=1) == 0).values
    #
    #     if both_value_mask.sum() < 5:
    #         continue
    #
    #     dwd_data = df_combine[both_value_mask].values[:, 0]
    #     net_data = df_combine[both_value_mask].values[:, 1]
    #
    #     # set indicator timeseries p0
    #     dwd_indi = np.zeros_like(dwd_data)
    #     dwd_indi[dwd_data == 0] = 1
    #
    #     net_indi = np.zeros_like(net_data)
    #     net_indi[net_data == 0] = 1
    #
    #     plt.scatter(distance,
    #                 np.corrcoef(dwd_indi, net_indi)[0, 1],
    #                 marker='o', c='black', s=2)
    #
    #     distance_list.append(distance)
    #     ik_list.append(np.corrcoef(dwd_indi, net_indi)[0, 1])
    #
    # idx = np.argsort(distance_list)
    # distance = np.array(distance_list)[idx]
    # ik_arr = np.array(ik_list)[idx]
    #
    # try:
    #     if np.where(ik_arr == np.max(ik_arr))[0][0] < 3:
    #         plt.close()
    #         continue
    #     else:
    #         plt.savefig(os.path.join(savepath,
    #                                  'ind_corr_p0_dwd_{}.png'.format(stn_id)))
    #         plt.close()
    # except (ValueError, IndexError):
    #     continue

    # # initialize lists
    # distance_list = []
    # ik_list = []
    #
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
    #
    # # calc p0 for all dwd stations
    # for dwd_id in range(dwd.root.name.shape[0]):
    #     # calc distance
    #     dwd_north = dwd.root.coord.northing[dwd_id]
    #     dwd_east = dwd.root.coord.easting[dwd_id]
    #
    #     distance = np.sqrt(
    #         (net_east - dwd_east) ** 2 + (net_north - dwd_north) ** 2)
    #
    #     df_dwd = pd.DataFrame(
    #         index=dwd.root.timestamps.isoformat.read(),
    #         columns=['dwd'],
    #         data=dwd.root.data[:, dwd_id])
    #
    #     df_dwd.index = pd.to_datetime(
    #         df_dwd.index.str.decode('utf-8'))
    #
    #     df_combine = pd.concat([df_dwd, df_netatmo], join='inner',
    #                            axis=1)
    #
    #     both_value_mask = (np.isnan(df_combine).sum(axis=1) == 0).values
    #
    #     if both_value_mask.sum() < 5:
    #         continue
    #
    #     dwd_data = df_combine[both_value_mask].values[:, 0]
    #     net_data = df_combine[both_value_mask].values[:, 1]
    #
    #     # set indicator timeseries p0
    #     dwd_indi = np.zeros_like(dwd_data)
    #     dwd_indi[dwd_data == 0] = 1
    #
    #     net_indi = np.zeros_like(net_data)
    #     net_indi[net_data == 0] = 1
    #
    #     ax[0].scatter(distance,
    #                 np.corrcoef(dwd_indi, net_indi)[0, 1],
    #                 marker='o', c='black', s=2)
    #
    #     distance_list.append(distance)
    #     ik_list.append(np.corrcoef(dwd_indi, net_indi)[0, 1])
    #
    #     # build edf
    #     try:
    #         dwd_edf = build_edf(np.sort(dwd_data))
    #         net_edf = build_edf(np.sort(net_data))
    #     except ZeroDivisionError:
    #         if both_value_mask.sum() == 0:
    #             # print(dwd_id, 'No time overlap')
    #             continue
    #
    #     # get 0.99
    #     try:
    #         dwd_tresh = sci.interp1d(dwd_edf, np.sort(dwd_data))(0.99)
    #         dwd_tresh1 = sci.interp1d(net_edf, np.sort(net_data))(0.99)
    #     except:
    #         continue
    #
    #     # set indicator timeseries
    #     dwd_indi_p99 = np.zeros_like(dwd_data)
    #     dwd_indi_p99[dwd_data >= dwd_tresh] = 1
    #
    #     net_indi_p99 = np.zeros_like(net_data)
    #     net_indi_p99[net_data >= dwd_tresh1] = 1
    #
    #     ax[1].scatter(distance,
    #                 np.corrcoef(dwd_indi_p99, net_indi_p99)[0, 1],
    #                 marker='o', c='black', s=2)
    #
    #
    #
    # idx = np.argsort(distance_list)
    # distance = np.array(distance_list)[idx]
    # ik_arr = np.array(ik_list)[idx]
    #
    # plt.show()
    # print()

# mail.status_mail(
#     subject='01_misplaced_filter.py finished %s' % datetime.datetime.now().strftime(
#         '%Y-%m-%d %H:%M:%S'))
