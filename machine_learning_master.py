
# import sub scripts
import data_load
import data_preprocessing
import model_building
import os
dir = os.path.dirname(__file__)


file_name = 'german_credit_car_fraud.ssv'
INPUT_DIR = dir + 'data/'
DATA_OUTPUT_DIR = dir + 'data/'
RESULTS_OUTPUT_DIR = dir + 'results/'

# run data load script
# if csv
initial_data = data_load.csvfile(INPUT_DIR, file_name, RESULTS_OUTPUT_DIR)
# if postgress
# initial_data = data_load.postgres(???)

# run preprocessing script
data_np_array, y = data_preprocessing.main(initial_data, DATA_OUTPUT_DIR)

# build initial models
model_building.modelsBuild(data_np_array, y)
# evaluation
# to add