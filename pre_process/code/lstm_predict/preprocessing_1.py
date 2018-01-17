#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:02:11 2018

@author: dingwangxiang
"""

# import your module here
import numpy as np
import pandas as pd

# (global) variable definition here
file_sets = [
        'DBID(1002089510)_INSTID(1)','DBID(2897570545)_INSTID(1)',
        'DBID(1227435885)_INSTID(1)','DBID(2949199900)_INSTID(1)',
        'DBID(1227435885)_INSTID(2)','DBID(3065831173)_INSTID(1)',
        'DBID(1254139675)_INSTID(1)','DBID(3111200895)_INSTID(1)',
        'DBID(1384807946)_INSTID(1)','DBID(3172835364)_INSTID(1)',
        'DBID(1624869053)_INSTID(1)','DBID(3204204681)_INSTID(1)',
        'DBID(1636599671)_INSTID(1)','DBID(3482311182)_INSTID(1)',
        'DBID(1636599671)_INSTID(2)','DBID(349165204)_INSTID(1)',
        'DBID(172908691)_INSTID(1)','DBID(3671658776)_INSTID(1)',
        'DBID(1855232979)_INSTID(1)','DBID(3671658776)_INSTID(2)',
        'DBID(1982696497)_INSTID(1)','DBID(3775482706)_INSTID(1)',
        'DBID(2031853600)_INSTID(1)','DBID(3775482706)_INSTID(2)',
        'DBID(2052255707)_INSTID(1)','DBID(4213264717)_INSTID(1)',
        'DBID(2238741707)_INSTID(1)','DBID(4215505906)_INSTID(1)',
        'DBID(2238741707)_INSTID(2)','DBID(4225426100)_INSTID(1)',
        'DBID(2328880794)_INSTID(1)','DBID(4291669003)_INSTID(1)',
        'DBID(2413621137)_INSTID(1)','DBID(4291669003)_INSTID(2)',
        'DBID(2612437783)_INSTID(1)','DBID(447326245)_INSTID(1)',
        'DBID(2644427317)_INSTID(1)','DBID(468957624)_INSTID(1)',
        'DBID(2707003786)_INSTID(1)','DBID(505574722)_INSTID(1)',
        'DBID(2762567375)_INSTID(1)','DBID(522516877)_INSTID(1)',
        'DBID(2768077198)_INSTID(1)','DBID(770699067)_INSTID(1)',
        'DBID(2778659381)_INSTID(1)','DBID(929227073)_INSTID(1)',
        'DBID(2778659381)_INSTID(2)','DBID(942093433)_INSTID(1)',
        'DBID(2802676787)_INSTID(1)','DBID(998852395)_INSTID(1)'
        ]

select_load = ['2080020','2080021','2080022','2080023','2080024','2080025','2080026',\
                  '2080027','2080028','2080029','2080030','2080031','2080032','2080033','2080034']

select_perf = ['2080040','2080041','2080042','2080043','2080044','2080045','2080046',\
               '2080047','2080048','2080049','2080050','2080051','2080052','2080053',\
               '2080054','2080055','2080056','2080057','2080058','2080059','2080060',\
                                       '2080061','2080062','2080063','2080064','2080065']

columns = ['Unnamed: 0', 'SnapId', 'StartTime', 'EndTime', '2080020', '2080021',
           '2080022', '2080023', '2080024', '2080025', '2080026', '2080027',
           '2080028', '2080029', '2080030', '2080031', '2080032', '2080033',
           '2080034', '2080040', '2080041', '2080042', '2080043', '2080044',
           '2080045', '2080046', '2080047', '2080048', '2080049', '2080050',
           '2080051', '2080052', '2080053', '2080054', '2080055', '2080056',
           '2080057', '2080058', '2080059', '2080060', '2080061', '2080062',
           '2080063', '2080064', '2080065', 'LoadScore', 'LoadLevel', 'PerfScore',
           'PerfLevel']

# class definition here

# function definition here

# look back dataset
def create_interval_dataset(dataset, lookback):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix. create_interval_dataset
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback):
        dataX.append(dataset[i:i+lookback])
        dataY.append(dataset[i+lookback])
    return np.asarray(dataX), np.asarray(dataY)

#  Imputation of missing values
def transform(data):
    for i in range(data.shape[1]):
        if pd.isnull(data[:,i]).sum() == 0:
            continue
        elif pd.isnull(data[:,i]).sum() == data.shape[0]:
            data[:,i] = [0] * data.shape[0]
        else:
            valid_val = []
            valid = [int, float]
            for j in range(data.shape[0]):
                if type(data[j][i]) in valid and not np.isnan(float(data[j][i])):
                    valid_val.append(data[j][i])
            if len(valid_val) == 0:
                mean = 0
            else:
                mean = sum(valid_val)/len(valid_val)
            for j in range(data.shape[0]):
                if np.isnan(float(data[j][i])):
                    data[j][i] = mean
    return data

# main program here
if __name__ == '__main__':
    for turn,file in enumerate(file_sets):
        db = pd.read_csv('../../csv/' + file + '.csv')
        print('open file ../../csv/' + file + '.csv')
        # Imputation of missing values
        db_values = db.values
        instance_db = db_values[:, 4:-5]
        for i in range(instance_db.shape[0]):
            for j in range(instance_db.shape[1]):
                # process the null
                if instance_db[i][j] == 'null':
                    # assign NaN to null for future process
                    instance_db[i][j] = np.nan
        transform(instance_db)
        db_values[:, 4:-5] = instance_db
        db = pd.DataFrame(db_values,columns = db.columns)