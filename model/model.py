#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for defining model classes for traditional machine learning algorithm
"""

import warnings
import inspect
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.kernel_approximation import Nystroem
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn import metrics
import json


class ModelTrainingWarning(UserWarning):
    pass


class BaseModel(object):
    """
    base model object for either classification or regression. Specific classes for specific use can be derived from this class
    """
    def __init__(self, model_type=None, high_prec_geohash_col="", date_col="", categorical_vars=None, numerical_vars=None, output_col="logistics_dropoff_distance", low_precision_level=5, do_kernel_transform=False, do_normalise_col=False, **kwargs):
        """
        :param model_type: sklearn model type or None
        :param high_prec_geohash_col: name of column with geohash code
        :param date_col: name of date column
        :param categorical_vars list of categorical variables
        :param numerical_vars list of numerical variables
        :param output_col: str, name of the output variable
        :param low_precision_level: int, lower precision level for geo code
        :param do_kernel_transform: Boolean variable whether to do Kernel approximation of features
        :param do_normalise_col: Boolean variable whether to scale column or not. (True scale feature using Robust scaler from sklearn; False no scaling

        """
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        if self.categorical_vars is None:
            self.categorical_vars = []
        self.categorical_vars = list(set(self.categorical_vars))

        if self.numerical_vars is None:
            self.numerical_vars = []
        if isinstance(self.numerical_vars, str):
            self.numerical_vars = [self.numerical_vars]
        self.numerical_vars = list(set(self.numerical_vars))

        self.feature_vars = self.categorical_vars + self.numerical_vars

        # building the overall feature extraction pipeline
        self.build_feature_extraction_pipeline()

        # building model pipeline
        if model_type is not None:
            self.best_model = self.build_model_pipeline()
        self.all_hyperparameters_list = [param for param in self.best_model.get_params() if "__" in param]

    def build_feature_extraction_pipeline(self):
        """
        method to build the feature extraction pipeline
        :return: feat extraction pipeline of type sklearn.pipeline.Pipeline
        """

        self.feat_ext = ColumnTransformer([("One_hot_encoder", OneHotEncoder(handle_unknown='ignore'), self.categorical_vars),
                                           ('impute_num_var', SimpleImputer(strategy='median'), self.numerical_vars)],
                                          remainder='drop', verbose=True, sparse_threshold=0.0)
        # scaling if specified
        if getattr(self, 'do_normalise_col', False):
            self.feat_ext = Pipeline([("feat_ext", self.feat_ext), ("scaling", RobustScaler())], verbose=True)

        # kernel transform if specified
        if getattr(self, 'do_kernel_transform', False):
            # if kernel transformation is specified to be done we do kernel transformation of the generated features to account for non-linearity
            self.feat_ext = Pipeline([("feat_ext", self.feat_ext), ("kernel_transform", Nystroem(n_components=1000))], verbose=True)
        return self.feat_ext

    def build_model_pipeline(self):
        """
        function build the overall model pipeline
        :return: model pipeline of type sklearn.pipeline.Pipeline
        """
        self.build_feature_extraction_pipeline()
        model_pipeline = Pipeline(
            [('feature_extraction_pipeline', self.feat_ext),
             ('reg', self.model_type)], verbose=True)
        return model_pipeline

    def fit(self, train_data, test_data=None, evaluation_metric='neg_mean_squared_log_error', **kwargs):
        """
        function to fit the model to training data.
        We can provide hyperparameter dictionary to select optimum hyperparameter values or else model is fitted with default values
        :param train_data: pandas dataframe to train the model
        :param test_data: pandas dataframe for model testing
        :param evaluation_metric: performance metric for cross validation fitting and model evaluation
        :param kwargs:
        :return: None
        """
        # preprocessing the data
        train_data = self.pre_process_data(train_data, self.date_col, self.high_prec_geohash_col, self.low_precision_level)

        # checking if the variables specified exist in the data set
        for var in self.feature_vars:
            assert var in train_data.columns, ("%s column not in the given dataset. Kindly ensure the spelling of each feature variable is correct" % var)

        assert self.output_col in train_data.columns, "Output column not in the given dataset. Kindly check for spelling errors"

        # getting the training input and output data
        if test_data is not None:
            test_data = self.pre_process_data(test_data, self.date_col, self.high_prec_geohash_col, self.low_precision_level)
            Y_test = test_data[self.output_col]
            X_test = test_data[self.feature_vars]

        else:
            print("no test data provided")
            X_test = None
            Y_test = None

        del test_data

        Y_train = train_data[self.output_col]
        X_train = train_data[self.feature_vars]
        del train_data

        tuning_hyperparameters_dict = {}
        args_dict = kwargs
        hyperparameter_dict = args_dict.get('hyperparameter_dict', {})
        # performance metric for cross validation and evaluation of the trained model.
        args_dict["evaluation_metric"] = evaluation_metric
        self.score_cal = metrics.get_scorer(args_dict["evaluation_metric"])
        complete_search = args_dict.get('complete_search', False)
        n_cpus = args_dict.get('n_cpus', -1)
        n_folds = args_dict.get('n_folds', 5)
        num_random_search_iter = args_dict.get('num_random_search_iter', 50)

        # getting hyperparameters dicitionary with actual name as per model pipeline.
        for param in hyperparameter_dict:
            for model_param in self.all_hyperparameters_list:
                if param == model_param.rsplit("__", 1)[1]:
                    tuning_hyperparameters_dict[model_param] = hyperparameter_dict[param]

        # check if hyper parameter dictionary is empty or not
        if not tuning_hyperparameters_dict:
            warnings.warn('No hyperparameters to train. fitting the model with default/pre-defined optimum values',
                          ModelTrainingWarning)
            self.best_model.fit(X_train, Y_train)
            if X_test is not None:
                print("calculating test score")
                self.best_score = self.score_cal(self.best_model, X_test, Y_test)
                print(self.best_score)

            return None

        if complete_search:
            print("Grid search for optimal hyper parameters")
            model_cross_val = GridSearchCV(self.best_model, param_grid=tuning_hyperparameters_dict,
                                           scoring=args_dict['evaluation_metric'], n_jobs=n_cpus,
                                           cv=n_folds, verbose=True)
        else:
            print("Randomized search for optimal hyper parameters")
            model_cross_val = RandomizedSearchCV(self.best_model, tuning_hyperparameters_dict,
                                                 scoring=args_dict['evaluation_metric'],
                                                 n_jobs=n_cpus, cv=n_folds,
                                                 n_iter=num_random_search_iter,
                                                 random_state=100, verbose=True)

        self.training_col = list(X_train.columns)
        model_cross_val.fit(X_train, Y_train)
        self.cross_val_result_summary = model_cross_val.cv_results_

        self.best_score = model_cross_val.best_score_
        self.best_model = model_cross_val.best_estimator_
        return None

    def save_model(self, model_loc):
        """
        method to save fitted model
        :param model_loc: directory to save the model
        :return:
        """
        if not os.path.isdir(os.path.split(model_loc)[0]) and os.path.split(model_loc)[0] != "":
            os.makedirs(os.path.split(model_loc)[0])
        joblib.dump(self.best_model, model_loc)

    def load_model(self, trained_model_loc):
        """
        method to load model from pickled file
        :param trained_model_loc:
        :return:
        """
        self.best_model = joblib.load(trained_model_loc)

    def predict_from_file(self, test_data_loc, pred_loc, trained_model_loc=None):
        try:
            with open(test_data_loc, 'r') as filein:
                test_data = pd.DataFrame.from_dict(json.load(filein))
        except Exception as e:
            print("Got following error while reading test data/n%s" % e)
            raise Exception("Cannot read the input test data. Please ensure the correct file "
                            "location including the directory and file name are both specified properly. "
                            "Please check the error message for more details")
        return self.predict(test_data, pred_loc, trained_model_loc)

    def predict(self, test_data, pred_loc=None, trained_model_loc=None):
        test_data = self.pre_process_data(test_data, self.date_col, self.high_prec_geohash_col, self.low_precision_level)
        test_data = test_data[self.feature_vars]
        # loading the model if the model location is specified or else using the built in model
        if trained_model_loc is not None:
            self.load_model(trained_model_loc)
        try:

            pred = pd.Series(self.best_model.predict(test_data))
            pred.name = "Predicted_logistics_dropoff_distance"
        except Exception as e:
            print("Got following error while predicting/n%s" % e)
            raise Exception("Error predicting the data. Please check the error message for more details")
        if pred_loc is not None:
            if not os.path.isdir(os.path.split(pred_loc)[0]) and os.path.split(pred_loc)[0] != "":
                os.makedirs(os.path.split(pred_loc)[0])
            pred.to_csv(pred_loc, header=True, index=False)
            return
        return pred

    @staticmethod
    def pre_process_data(input_df, date_col, higher_geohash_col_name, low_prec_level=5):
        input_df = input_df.copy()
        input_df[date_col] = pd.to_datetime(input_df[date_col])
        input_df["day_of_week"] = input_df[date_col].dt.day_name()
        input_df["hour_of_day"] = input_df[date_col].dt.hour
        input_df["geohash_code_precision%d" % low_prec_level] = input_df[higher_geohash_col_name].str[:low_prec_level]
        return input_df
