# import sub scripts and libraries
import logging
import datetime
import data_load
import data_preprocessing
import model_building   
import utils
import pprint as pp # to make log entries nicer to read
# get the settings for the run
import settings


#Setup the logger
utils.setLog(settings.LOGGING_FILE)
logger=logging.getLogger('Master')

#Log start of Full process
utils.logInfoTime(logger, 'Started')


# run data load script
logger.info('--------------------------------- Data Load -----------------------------------')
utils.logInfoTime(logger, 'Started Data Load')
# initial_data = data_load.psqlLoad(settings.INPUT_TABLE, settings.INPUT_SCHEMA, columns='*')
initial_data = data_load.csvfile(settings.INPUT_DIR, settings.file_name, settings.RESULTS_OUTPUT_DIR)
utils.logInfoTime(logger, 'Finished Data Load')

# run preprocessing script
logger.info('--------------------------------- Data Preprocessing -----------------------------------')
utils.logInfoTime(logger, 'Started Pre-Processing')
data_np_array, y_np_array, var_results = data_preprocessing.main(initial_data)
utils.logInfoTime(logger, 'Finished Pre-Processing')

# build models
logger.info('--------------------------------- Model Building -----------------------------------')
utils.logInfoTime(logger, 'Started Model Building')
model_building.modelsBuild(data_np_array, y_np_array, var_results, logger)
utils.logInfoTime(logger, 'Finished Model Building')

utils.logInfoTime(logger, 'Finished')