# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:08:03 2024
@author: RJL
"""


import numpy as np
import os
import hdf5storage


def load_data(matpath = 'Z:/SID-模型训练-24h-1km-add/创建数据集/'):

    # Read mat file
    input_train_mat = hdf5storage.loadmat(matpath+'SID_input_train')
    outputdata_train_mat = hdf5storage.loadmat(matpath+'SID_output_train')
    input_valid_mat = hdf5storage.loadmat(matpath+'SID_input_valid')
    outputdata_valid_mat = hdf5storage.loadmat(matpath+'SID_output_valid')
    maxminmeanstd_mat = hdf5storage.loadmat(matpath+'maxminstdmean')

    input_train = input_train_mat['SID_input_train']
    output_train = outputdata_train_mat['SID_output_train']
    
    input_valid = input_valid_mat['SID_input_valid']
    output_valid = outputdata_valid_mat['SID_output_valid']
    
    m, n = input_train.shape
    Inputdata_train = np.empty((m, 10))
    
    Inputdata_train[:, 0] = (input_train[:, 0]-maxminmeanstd_mat['wind_u_min']) / \
        (maxminmeanstd_mat['wind_u_max']-maxminmeanstd_mat['wind_u_min'])
    Inputdata_train[:, 1] = (input_train[:, 1]-maxminmeanstd_mat['wind_v_min']) / \
        (maxminmeanstd_mat['wind_v_max']-maxminmeanstd_mat['wind_v_min'])
    Inputdata_train[:, 2] = (input_train[:, 2]-maxminmeanstd_mat['geocurrent_u_min']) / \
        (maxminmeanstd_mat['geocurrent_u_max']-maxminmeanstd_mat['geocurrent_u_min'])
    Inputdata_train[:, 3] = (input_train[:, 3]-maxminmeanstd_mat['geocurrent_v_min']) / \
        (maxminmeanstd_mat['geocurrent_v_max']-maxminmeanstd_mat['geocurrent_v_min'])
    Inputdata_train[:, 4] = (input_train[:, 4]-maxminmeanstd_mat['hice_min']) / \
        (maxminmeanstd_mat['hice_max']-maxminmeanstd_mat['hice_min'])
    Inputdata_train[:, 5] = (input_train[:, 5]-maxminmeanstd_mat['slp_min']) / \
        (maxminmeanstd_mat['slp_max']-maxminmeanstd_mat['slp_min'])
    Inputdata_train[:, 6] = (input_train[:, 6]-maxminmeanstd_mat['t2m_min']) / \
        (maxminmeanstd_mat['t2m_max']-maxminmeanstd_mat['t2m_min'])
    Inputdata_train[:, 7] = (input_train[:, 7]-maxminmeanstd_mat['lat_min']) / \
        (maxminmeanstd_mat['lat_max']-maxminmeanstd_mat['lat_min'])
    Inputdata_train[:, 8] = (input_train[:, 8]-maxminmeanstd_mat['lon_min']) / \
        (maxminmeanstd_mat['lon_max']-maxminmeanstd_mat['lon_min'])
    Inputdata_train[:, 9] = (input_train[:, 9]-maxminmeanstd_mat['fgimg_min']) / \
        (maxminmeanstd_mat['fgimg_max']-maxminmeanstd_mat['fgimg_min'])
    
    outputdata_train = output_train
    
    m, n = input_valid.shape
    Inputdata_valid = np.empty((m, 10))
    
    Inputdata_valid[:, 0] = (input_valid[:, 0]-maxminmeanstd_mat['wind_u_min']) / \
        (maxminmeanstd_mat['wind_u_max']-maxminmeanstd_mat['wind_u_min'])
    Inputdata_valid[:, 1] = (input_valid[:, 1]-maxminmeanstd_mat['wind_v_min']) / \
        (maxminmeanstd_mat['wind_v_max']-maxminmeanstd_mat['wind_v_min'])
    Inputdata_valid[:, 2] = (input_valid[:, 2]-maxminmeanstd_mat['geocurrent_u_min'])/(
        maxminmeanstd_mat['geocurrent_u_max']-maxminmeanstd_mat['geocurrent_u_min'])
    Inputdata_valid[:, 3] = (input_valid[:, 3]-maxminmeanstd_mat['geocurrent_v_min'])/(
        maxminmeanstd_mat['geocurrent_v_max']-maxminmeanstd_mat['geocurrent_v_min'])
    Inputdata_valid[:, 4] = (input_valid[:, 4]-maxminmeanstd_mat['hice_min']) / \
        (maxminmeanstd_mat['hice_max']-maxminmeanstd_mat['hice_min'])
    Inputdata_valid[:, 5] = (input_valid[:, 5]-maxminmeanstd_mat['slp_min']) / \
        (maxminmeanstd_mat['slp_max']-maxminmeanstd_mat['slp_min'])
    Inputdata_valid[:, 6] = (input_valid[:, 6]-maxminmeanstd_mat['t2m_min']) / \
        (maxminmeanstd_mat['t2m_max']-maxminmeanstd_mat['t2m_min'])
    Inputdata_valid[:, 7] = (input_valid[:, 7]-maxminmeanstd_mat['lat_min']) / \
        (maxminmeanstd_mat['lat_max']-maxminmeanstd_mat['lat_min'])
    Inputdata_valid[:, 8] = (input_valid[:, 8]-maxminmeanstd_mat['lon_min']) / \
        (maxminmeanstd_mat['lon_max']-maxminmeanstd_mat['lon_min'])
    Inputdata_valid[:, 9] = (input_valid[:, 9]-maxminmeanstd_mat['fgimg_min']) / \
        (maxminmeanstd_mat['fgimg_max']-maxminmeanstd_mat['fgimg_min'])
    
    outputdata_valid = output_valid
    
    return Inputdata_train,outputdata_train,Inputdata_valid,outputdata_valid