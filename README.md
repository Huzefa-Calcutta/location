# Predicting Logistics Drop-off Distance
## 1. Introduction
This is a Python based repo for predicting drop off distance for food delivery. 
This repo could be also used for any regression problems and can even be extended to classification with slight modification
## 2. Training Framework
We use scikit learn library for this exercise, where we define attributes and methods to encapsulate the entire end to training and inference process, right from data preprocessing, feature engineering and model fitting and model prediction, saving and loading of trained models. The BaseModel class takes the following attributes:

model_type: sklearn model type or None
high_prec_geohash_col: name of column with geohash code
date_col: name of date column
categorical_vars list of categorical variables
numerical_vars list of numerical variables
output_col: name of the output variable
low_precision_level: lower precision level for geo code
do_kernel_transform: Boolean variable whether to do Kernel approximation of features
do_normalise_col: Boolean variable whether to scale column or not. (True scale feature using Robust scaler from sklearn. if False no sclaing is done
Additional keyword arguments could also be passed. The BaseModel class has best_model attribute which is essentially sklearn pipeline including feature processing and model. Depending on the alogorithm you choose, the feature processing steps could be customised. This Class can also be used for Classification kind of problems.
 
The current model in production uses approach one. 
- Python Files
  1. `model/model.py` Defines the generic model class which can be used for either regression or even classification problems.
  2. `inference.py` Inference script. It takes config file as input
  3. `utils/keras_data_gen.py` Util script for generating data for training and inference 
  4. `utils/multi_view_data_loader.py` script for defining generator class for multiple images as input
- Config files
`model_train.config` A config, which is consumed by the jupyter Notebook Model_training.ipynb,  containing spec for training the model such as data location, name of features, model training parameters
`model_inference.cfg` A config, which is consumed by the python script inference.py,  containing spec for prediction such as test data location, predicted data location, model location and name of columns data to be used.
  
The current model in production uses approach one. 

## 3. folder structure of data and Model
```
# General Data Structure with explicit validation data
data/
    -{train_file}.json
    -{test_file}.json
    -{predicted_file}.csv
    -model/
          {model_file}.pkl
# Example
data/
    -example_train.json
    -example_test.json
    -example_pred.csv
    -model/
          example_model.pkl
    
```

## 4. Config files
You should see a file called `model_train.cfg` in your root directory of the repo. You need to modify some config as shown in the below example according your model needs.
```
[input]
train_data = data/location_task_no_nulls.json # location of training data

[model_specs]
output_col = logistics_dropoff_distance # name of output column
date_col = created_timestamp_local # name of date column
high_geohash_precision_table = delivery_geohash_precision8 # name of column containing high precision code geohash
categorical_vars = day_of_week,hour_of_day,geohash_code_precision5 # name of categorical columns separated by commas with no space 
numerical_vars = has_instruction,has_phone_number,has_call_instruction,has_leave_instruction,has_lift_instruction,has_lobby_instruction,has_gate_instruction,has_knock_instruction,has_bell_instruction,order_value,order_items_count # name of numerical variables separated by commas with no spaces

[model]
dir = data/model/ # directory to save the trained model

```
Similarly we can define the model_inference.cfg file.

## 5. Setting up the system
1. Installing miniconda on Linux and macOS
  ```
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p ~/miniconda3
  rm -rf miniconda.sh
  ~/miniconda3/bin/conda init bash
  ~/miniconda3/bin/conda init zsh  
  ```
2. Create python env
  ```
  conda create --name location python=3.9
  conda activate location
  ```

3. Installing Python packages
  cd to the root of this repo
  ```
  cd [path_repo]
  ```
  running pip install
  ```
  pip install -r requirements.txt
  ```
## 6. Training the model 
Use the steps as specified in the jupyter notebook Model_train.ipynb
You can experiment with different model algorithms

### 7. Inference 
If the system is not step up, please follow steps as mentioned above.
1. Define the model_inference.cfg file. Specify the path of model and test data properly 
2. In the terminal, go to root directory
  ```
  cd [path_repo]
  ```
3. Run inference python script
  ```
  python inference.py
  ```