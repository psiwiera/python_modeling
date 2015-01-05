
# import sub scripts
import data_load
import data_preprocessing
import model_building

file_name = 'german_credit_car_fraud.ssv'
INPUT_DIR = 'xxx'
DATA_OUTPUT_DIR = 'data/'
RESULTS_OUTPUT_DIR = 'results/'

# run data load script
# if csv
initial_data = data_load.csvfile(INPUT_DIR, file_name, RESULTS_OUTPUT_DIR)
# if postgress
# initial_data = data_load.postgres(???)
# run preprocessing script
processed_data_np_array = data_preprocessing.main(initial_data, DATA_OUTPUT_DIR)
# build initial models
model_building.pipeline(processed_data_np_array)
# evaluation
# to add