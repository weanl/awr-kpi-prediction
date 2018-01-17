#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:10:46 2017

@author: dingwangxiang
"""

# import your module here
import pandas as pd
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout
# from sklearn.preprocessing import MinMaxScaler
from trainingset_selection import TrainingSetSelection
from keras.models import load_model, save_model
from keras.utils import plot_model
#from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from preprocessing import get_ids_and_files_in_dir, percentile_remove_outlier, MinMaxScaler, NormalDistributionScaler, binning_date_y
import os

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
        'DBID(2802676787)_INSTID(1)','DBID(998852395)_INSTID(1)',
        ]

# class definition here
# function definition here
def create_interval_dataset(dataset, lookback):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback):
        dataX.append(dataset[i:i+lookback])
        dataY.append(dataset[i+lookback])
    return np.asarray(dataX), np.asarray(dataY)


def main():
    print("a1")
    training_set_dir = "../../time_series_50_to_1"                             # "/your_local_path/RNN_prediction_2/cluster/train"
    output_dir = "./cluster_lstm_model"                                        # "/your_local_path/RNN_prediction_2/cluster_lstm_model"
    training_set_id_range = (1002089510, 1002089510)
    training_set_length = 50
    dense_layer = 2
    model_file_prefix = 'model'
    model_save_dir = output_dir + "/" + model_file_prefix
    training_set_regx_format = "(DBID)\((\d+)\)_INSTID\([1]\)_perf.csv"        # "cluster-(\d+)\.csv"
    print("a2")
    obj_NN = NeuralNetwork(output_dir=output_dir,
                           training_set_dir=training_set_dir,
                           model_save_dir=model_save_dir,
                           model_file_prefix=model_file_prefix,
                           training_set_id_range=training_set_id_range,
                           training_set_length=training_set_length,
                           dense_layer=dense_layer)
    # record program process printout in log file
    """
    stdout_backup = sys.stdout
    log_file_path = output_dir + "/NN_model_running_log.txt"
    log_file_handler = open(log_file_path, "w")
    print ("Log message could be found in file: {}".format(log_file_path))
    sys.stdout = log_file_handler
    """
    # check if the training set directory is empty. If so, run the training set selection
    if not os.listdir(obj_NN.training_set_dir):
        print ("Training set files not exist! Run trainingSetSelection.trainingSetGeneration to generate them! Start running generating training set files...")
        trainingSetObj = TrainingSetSelection(min_purchase_count=4)
        trainingSetObj.trainingset_generation(outdir=obj_NN.training_set_dir)
        print ("Training set file generation done! They are store at %s directory!".format(obj_NN.training_set_dir))
    print ("Train NN model and test!")
    obj_NN.model_train_predict_test(override=False, input_file_regx=training_set_regx_format)
    print ("Models and their parameters are stored in {}".format(obj_NN.model_save_dir))
    # close log file
    """
    log_file_handler.close()
    sys.stdout = stdout_backup
    """

# main program here
if  __name__ == '__main__':
    main()
    """
    csv_file_name = '../../time_series_one/' + file_sets[0] + '_perf'+'.csv'
    df = pd.read_csv(csv_file_name)    
    dataset_init = np.asarray(df).reshape(-1)    # if only 1 column
    dataX, dataY = create_interval_dataset(dataset_init, lookback=50)    # look back if the training set sequence length
    df_new = pd.DataFrame(dataX, columns = ['X' + str(i) for i in range(1, 1+dataX.shape[1])])
    df_new['Y'] = pd.Series(dataY,index = df_new.index)
    df_new.to_csv(path_or_buf = '../../time_series_50_to_1/' + file_sets[0] + '_perf'+'.csv', index=False)
    """

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    