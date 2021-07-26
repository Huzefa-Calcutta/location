#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for doing inference
"""

import os
import datetime
import configparser
from model.model import *


if __name__ == '__main__':
    # loading the config file with info about location of test data and model
    cfgParse = configparser.ConfigParser()
    cfgParse.read("model_test.cfg")

    # location of model storage
    model_loc = cfgParse.get("model", "path")

    # location of test data
    test_data_loc = cfgParse.get("data", "test_data_loc")
    # directory where predictions have to be stored
    prediction_loc = cfgParse.get('data', 'pred_data_loc')
    # creating the directory for storing the predicted values if it is not there.
    if not os.path.isdir(os.path.split(prediction_loc)[0]) or os.path.split(prediction_loc)[0]== "" or os.path.split(prediction_loc)[0]== ".":
        os.makedirs(os.path.split(prediction_loc)[0])
    # Reading the model specs i.e. features defined for trained model
    output_col = str(cfgParse.get("model_specs", "output_col"))
    high_geohash_precision = cfgParse.get("model_specs", "high_geohash_precision_table")
    date_col = cfgParse.get("model_specs", "date_col")
    categorical_vars = [col for col in cfgParse.get("model_specs", "categorical_vars").split(",")]
    numerical_vars = [col for col in cfgParse.get("model_specs", "numerical_vars").split(",")]

    prediction_time_st = datetime.datetime.now()
    # creating classifier instance
    model = BaseModel(None, high_geohash_precision, date_col,
                             categorical_vars, numerical_vars, output_col)

    model.predict_from_file(model_loc, test_data_loc, prediction_loc)

    prediction_time_end = datetime.datetime.now()
    prediction_time = (prediction_time_end - prediction_time_st).total_seconds() / 60.0
    print("Time required for prediction for random forest model is %.3f minutes" % prediction_time)
