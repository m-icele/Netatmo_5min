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
import glob
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
import pyproj

sys.path.append(r'D:\py-sele')
import mail

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# settings
# ==============================================================================
# get working directory
cwd = os.getcwd()
netatmo_folder = os.path.dirname(cwd)

# path to dwd
path_dwd = os.path.join(netatmo_folder, r"01_data/DWD_5min_to_1hour.h5")

# path to netatmo data
path_netatmo = os.path.join(netatmo_folder,
                            # r"01_data/netatmo_Germany_5min_to_1hour_filter_00.h5")
                                r"01_data/netatmo_Germany_5min_to_1hour_2014_2020.h5")

savepath = os.path.join(netatmo_folder, r'03_results/01_misplaced_filter')

# # path to dwd
# path_dwd = r"D:\bwsyncandshare\Netatmo_DWD\03_dwd\DWD_5min_to_1hour.h5"
#
# # path to netatmo data
# path_netatmo = (r"D:\bwsyncandshare\Netatmo_DWD\01_netatmo"
#                 r"\netatmo_Germany_5min_to_1hour_2014_2020.h5")
# # path_netatmo = (r"D:\bwsyncandshare\Netatmo_DWD\01_netatmo"
# #                 r"\netatmo_Germany_5min_to_1hour_filter_00.h5")
#
# path to station scans
path_scans = (r'D:\Netatmo_5min\01_data\station_scans')
#
# savepath = (r'D:\Netatmo_5min\03_results\01_misplaced_filter')

multi_processing = True


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


def calc_p0_over_distance(net_east, net_north, dwd_east, dwd_north, df_combine):
    '''This function calculates the distance and p0 indicator correlation for
    the two given stations.'''
    # calc distance
    distance = np.sqrt(
        (net_east - dwd_east) ** 2 + (net_north - dwd_north) ** 2)

    both_value_mask = (np.isnan(df_combine).sum(axis=1) == 0).values

    if both_value_mask.sum() < 5:
        return distance, np.nan

    dwd_data = df_combine[both_value_mask].values[:, 0]
    net_data = df_combine[both_value_mask].values[:, 1]

    # set indicator timeseries p0
    dwd_indi = np.zeros_like(dwd_data)
    dwd_indi[dwd_data == 0] = 1

    net_indi = np.zeros_like(net_data)
    net_indi[net_data == 0] = 1

    return distance, np.corrcoef(dwd_indi, net_indi)[0, 1]


def calc_p99_over_distance(net_east, net_north, dwd_east, dwd_north, df_combine):
    '''This function calculates the distance and p99 indicator correlation for
    the two given stations.'''
    # calc distance
    distance = np.sqrt(
        (net_east - dwd_east) ** 2 + (net_north - dwd_north) ** 2)

    both_value_mask = (np.isnan(df_combine).sum(axis=1) == 0).values

    if both_value_mask.sum() < 5:
        return distance, np.nan

    dwd_data = df_combine[both_value_mask].values[:, 0]
    net_data = df_combine[both_value_mask].values[:, 1]

    # build edf
    try:
        dwd_edf = build_edf(np.sort(dwd_data))
        net_edf = build_edf(np.sort(net_data))
    except ZeroDivisionError:
        if both_value_mask.sum() == 0:
            # print(dwd_id, 'No time overlap')
            return distance, np.nan

    # get 0.99
    try:
        dwd_tresh = sci.interp1d(dwd_edf, np.sort(dwd_data))(0.99)
        dwd_tresh1 = sci.interp1d(net_edf, np.sort(net_data))(0.99)
    except:
        return distance, np.nan

    # set indicator timeseries
    dwd_indi_p99 = np.zeros_like(dwd_data)
    dwd_indi_p99[dwd_data >= dwd_tresh] = 1

    net_indi_p99 = np.zeros_like(net_data)
    net_indi_p99[net_data >= dwd_tresh1] = 1

    return distance, np.corrcoef(dwd_indi_p99, net_indi_p99)[0, 1]


def convert_coords_fr_wgs84_to_utm32_(epgs_initial_str, epsg_final_str,
                                      first_coord, second_coord):
    """
    Purpose: Convert points from one reference system to a second
    --------
        In our case the function is used to transform WGS84 to UTM32
        (or vice versa), for transforming the DWD and Netatmo station
        coordinates to same reference system.

        Used for calculating the distance matrix between stations

    Keyword argument:
    -----------------
        epsg_initial_str: EPSG code as string for initial reference system
        epsg_final_str: EPSG code as string for final reference system
        first_coord: numpy array of X or Longitude coordinates
        second_coord: numpy array of Y or Latitude coordinates

    Returns:
    -------
        x, y: two numpy arrays containing the transformed coordinates in
        the final coordinates system
    """
    # initial_epsg = pyproj.Proj(init=epgs_initial_str)
    initial_epsg = pyproj.Proj(epgs_initial_str)
    # final_epsg = pyproj.Proj(init=epsg_final_str)
    final_epsg = pyproj.Proj(epsg_final_str)
    x, y = pyproj.transform(initial_epsg, final_epsg,
                            first_coord, second_coord)
    return x, y


def calc_p0_for_given_coordinates(args):
    (dwd_tree, net_north, net_east, savepath, path_dwd, df_netatmo,
     stn_id, stn_name) = args

    print(stn_id)

    dwd = tables.open_file(path_dwd, 'r')

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

        distance, p0_ic = calc_p0_over_distance(net_east, net_north, dwd_east,
                                                dwd_north,
                                                df_combine)

        plt.scatter(distance,
                    p0_ic,
                    marker='o', c='black', s=2)

        distance_list.append(distance)
        ik_list.append(p0_ic)

    idx = np.argsort(distance_list)
    distance_arr = np.array(distance_list)[idx]
    ik_arr = np.array(ik_list)[idx]

    try:
        if np.where(ik_arr == np.max(ik_arr))[0][0] < 3:
            plt.close()
            dwd.close()
            return
    except (ValueError, IndexError):
        plt.close()
        dwd.close()
        return

    # check available coordinates
    scan_files = glob.glob(os.path.join(path_scans, '*.csv'))
    coords_available = False
    station_moved = False
    for scan in scan_files:
        mac_addresses = pd.read_csv(scan, sep=';', index_col=0,
                                    dtype={'lon': np.float64,
                                           'lat': np.float64})

        try:
            if coords_available:
                if (mac_addresses.loc[stn_name.decode(), 'lon'] == init_lon) & (
                        mac_addresses.loc[
                            stn_name.decode(), 'lat'] == init_lat):
                    continue
                else:
                    init_etrs89 = convert_coords_fr_wgs84_to_utm32_(
                        '+init=epsg:4326', '+init=epsg:25832',
                        init_lon, init_lat)

                    etrs89 = convert_coords_fr_wgs84_to_utm32_(
                        '+init=epsg:4326', '+init=epsg:25832',
                        mac_addresses.loc[stn_name.decode(), 'lon'],
                        mac_addresses.loc[stn_name.decode(), 'lat'])

                    distance = np.sqrt((init_etrs89[0] - etrs89[0]) ** 2 + (
                            init_etrs89[1] - etrs89[1]) ** 2)

                    if distance < 50:
                        continue
                    else:
                        print('station moved')
                        station_moved = True

            else:
                init_lon = mac_addresses.loc[stn_name.decode(), 'lon']
                init_lat = mac_addresses.loc[stn_name.decode(), 'lat']
                coords_available = True

            # print(os.path.basename(scan)[:10],
            #       mac_addresses.loc[stn_name.decode(), 'lon'],
            #       mac_addresses.loc[stn_name.decode(), 'lat'])
        except KeyError:
            continue

    # divide in single years
    for year in df_netatmo.index.year.unique():
        if df_combine.index[0] > pd.Timestamp('{}-04-01 01:00:00'.format(year)):
            start_index = df_combine.index[0]
        else:
            start_index = pd.Timestamp('{}-04-01 01:00:00'.format(year))

        if df_combine.index[-1] < pd.Timestamp(
                '{}-11-01 01:00:00'.format(year)):
            end_index = df_combine.index[-1]
        else:
            end_index = pd.Timestamp('{}-11-01 01:00:00'.format(year))

        if start_index >= end_index:
            continue

        # select year
        df_year = df_netatmo.loc[start_index:end_index]

        # initialize lists
        distance_list = []
        ik_list = []

        # for dwd_id in selected_dwds:
        for dwd_id in  range(dwd.root.name.shape[0]):
            dwd_north = dwd.root.coord.northing[dwd_id]
            dwd_east = dwd.root.coord.easting[dwd_id]

            df_dwd = pd.DataFrame(index=dwd.root.timestamps.isoformat.read(),
                                  columns=['dwd'],
                                  data=dwd.root.data[:, dwd_id])

            df_dwd.index = pd.to_datetime(
                df_dwd.index.str.decode('utf-8'))

            df_combine_temp = pd.concat([df_dwd, df_year], join='inner',
                                   axis=1)

            distance, p0_ic = calc_p0_over_distance(net_east, net_north,
                                                    dwd_east,
                                                    dwd_north,
                                                    df_combine_temp)

            plt.scatter(distance,
                        p0_ic,
                        marker='o', c='black', s=2)

            distance_list.append(distance)
            ik_list.append(p0_ic)

        plt.ylim([0, 1])
        plt.xlim(left=0)
        plt.ylabel('indicator correlation p0 [-]')
        plt.xlabel('distance between stations [m]')
        plt.title('{} {}'.format(stn_id, year))
        plt.grid()
        plt.tight_layout()

        plt.savefig(os.path.join(savepath, '{}_{}.png'.format(stn_id, year)))
        plt.close()

    # TODO: IK p0 als funktion schreiben

    if not station_moved:
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

        distance, p0_ic = calc_p0_over_distance(net_east, net_north,
                                                dwd_east,
                                                dwd_north,
                                                df_combine)

        ax[0].scatter(distance,
                      p0_ic,
                      marker='o', c='black', s=2)

        distance_list.append(distance)
        ik_list.append(p0_ic)

        distance, p99_ic = calc_p99_over_distance(net_east, net_north,
                                                dwd_east,
                                                dwd_north,
                                                df_combine)

        ax[1].scatter(distance,
                      p99_ic,
                      marker='o', c='black', s=2)

        ax[0].set_ylim([0, 1])
        ax[0].set_xlim(left=0)
        ax[1].set_xlim(left=0)
        ax[0].set_ylabel('indicator correlation p0/p99 [-]')
        ax[0].set_xlabel('distance between stations [m]')
        ax[1].set_xlabel('distance between stations [m]')
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title('{} p0'.format(stn_id))
        ax[0].set_title('{} p99'.format(stn_id))
        plt.tight_layout()

    plt.savefig(os.path.join(savepath,
                             '{}_all.png'.format(
                                 stn_id)))
    plt.close()

    dwd.close()
    return


# ==============================================================================
# process
# ==============================================================================
def process_manager():
    # open data sets
    netatmo_hf = tables.open_file(path_netatmo, 'r')
    dwd = tables.open_file(path_dwd, 'r')

    # set up dwd tree
    dwd_coords = np.vstack((dwd.root.coord.northing[:],
                            dwd.root.coord.easting[:])).T

    dwd_tree = ssp.cKDTree(dwd_coords)

    if multi_processing:
        # initialize multiprocessing
        my_pool = mp.Pool(10)
        args = ()

    # 1. identify possible misplaced stations
    for stn_id, stn_name in enumerate(netatmo_hf.root.name.read()):
        if stn_id < 415:
            continue

        if stn_id > 500:
            continue

        print(stn_id, stn_name)

        # get data for current station
        df_netatmo = pd.DataFrame(
            index=netatmo_hf.root.timestamps.isoformat.read(),
            columns=['netatmo'],
            data=netatmo_hf.root.data[:, stn_id])[1:-1]

        df_netatmo.index = pd.to_datetime(df_netatmo.index.str.decode('utf-8'))
        #######################################
        # check NaNs at beginning or end
        net_value_mask = (np.isnan(df_netatmo).sum(axis=1) == 0).values
        df_n_mask_end = (
                net_value_mask.cumsum() == net_value_mask.cumsum()[-1]).sum()
        df_n_mask_start = np.where(net_value_mask.cumsum() == 0)[0][-1]

        # cut timeseries
        df_netatmo = df_netatmo.iloc[int(df_n_mask_start + 1):]
        df_netatmo = df_netatmo.drop(df_netatmo.tail(df_n_mask_end).index)
        #################################################

        # get coordinates
        net_north = netatmo_hf.root.coord.northing[stn_id]
        net_east = netatmo_hf.root.coord.easting[stn_id]

        if multi_processing:
            args += (
                (dwd_tree, net_north, net_east, savepath, path_dwd, df_netatmo,
                 stn_id, stn_name),)
        else:
            args = (dwd_tree, net_north, net_east, savepath, path_dwd,
                    df_netatmo, stn_id, stn_name)
            calc_p0_for_given_coordinates(args)

    netatmo_hf.close()
    dwd.close()

    if multi_processing:
        my_pool.map(calc_p0_for_given_coordinates, args)

        my_pool.close()
        my_pool.join()


if __name__ == '__main__':
    start = datetime.datetime.now()
    process_manager()

    print(datetime.datetime.now() - start)

    mail.status_mail(
        subject='01_misplaced_filter.py finished %s' % datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'))
