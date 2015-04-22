# Script to execute previously build models on a new dataset 
# import sub scripts
import logging
import data_load
import data_preprocessing
import data_preprocessing_functions as ppfuncs 
import model_building   
import os
dir = os.path.dirname(__file__)
import pprint as pp # to make log entries nicer to read
from sklearn.externals import joblib
import json
import pandas as pd
from sklearn.metrics import roc_curve
import datetime
import pprint as pp # to make log entries nicer to read
import model_building_functions as modFuncs
import utils
import settings
import data_preprocessing as pproc
import model_execution_functions as mexec

#Setup the logger
logger = utils.setLog(settings.logging_file_exec,logtype='Exec') 
logger = logging.getLogger('Exec.model_execution')
    
logger.info('Started %s', datetime.datetime.now().time().isoformat())
logger.info('--------------------------------- Data Load -----------------------------------')
data, initial_data = mexec.loadDataToScore(settings.INPUT_DIR,settings.FILE_NAME_EXEC,settings.RESULTS_OUTPUT_DIR)

logger.info('--------------------------------- Apply Pre-Processing Steps-----------------------------------')
data, var_results = pproc.main(data,execute=True)

if settings.feature_selection:
    data,var_results = mexec.applyFeatureSelection(data,settings.MODELS_OUTPUT_DIR)
    
logger.info('--------------------------------- Apply Model -----------------------------------')
output_df,output,clf = mexec.applyModel(settings.model_pickle_file,data,initial_data,settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR + '/test_data.pkl')

if settings.decision_tree==True:
    var_list = mexec.makePrettyTree(settings.dot_file,data,var_results,settings.RESULTS_OUTPUT_DIR)
else:
    var_list = list(set(initial_data.columns))


# need to make sure we have the probability of bad in the variable list
var_list.append('prob_class_1')
var_list = list(set(var_list))
# reduce the output datafrom to the final columns
output_df=output_df[var_list]

# rename the columns - Postgres has a limit of 63 characters for a table/column name
output_df = mexec.renameCols(output_df,var_list)

# check the final dataframe after all the merges
#mexec.qaFinalDF(output_df,initial_data,output)

logger.info('--------------------------------- Write scored data back to db -----------------------------------')
# drop previous table and index
#data_load.run_sql_string("""DROP TABLE IF EXISTS """+settings.input_schema_exec+"""."""+settings.output_table_exec+""" CASCADE;""",'Exec')
#data_load.run_sql_string("""DROP INDEX IF EXISTS """+settings.input_schema_exec+""".ix_"""+settings.input_schema_exec+"""_"""+settings.output_table_exec+"""_pd_index CASCADE;""",'Exec')
# write to db
#data_load.psqlBulkWrite(output_df, settings.output_table_exec, settings.input_schema_exec,'Exec')

output_df.to_csv(settings.output_file)

# Now run any rules or anomaly detection and so on
logger.info('--------------------------------- Apply Rules -----------------------------------')
#mexec.applyRules(settings.input_schema_exec,settings.output_table_exec)


print('model execution complete')
