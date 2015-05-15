import os
dir = os.path.dirname(__file__)

# set the values for the input data
#INPUT_TABLE = 'risk_scoring_master_vector'
#INPUT_SCHEMA = 'analytics_sandpit'
#file_name = 'german_credit_car_fraud.ssv'
file_name = 'credit_risk_kaggle.csv'
INPUT_DIR = dir + '/data/'


# set the output directories
# Note in general this doesn't need to be changed
DATA_OUTPUT_DIR = dir + '/data/'
RESULTS_OUTPUT_DIR = dir + '/results/'
MODELS_OUTPUT_DIR = dir + '/models/'
LOGGING_FILE = dir + '/results_log.log'

# dictionary listing what models to include in the run and which not
models = {
          'Decision Tree':True
          ,'Logistic Regression':True
          ,'Random Forest':True
          ,'Logistic Regression Grid Search':False
          ,'Decision Tree Grid Search':False
          ,'Random Forest Grid Search':False
          ,'SGD':False
         }
         
feature_selection=True
variable_scaling=False #current best model without scaling.

# Set the values for the model execution
# Set some variable values
input_table_exec = 'risk_scoring_ids_to_score'
input_schema_exec = 'analytics_sandpit'
output_table_exec = 'risk_scoring_output'
#FILE_NAME_EXEC = 'german_credit_car_fraud_testcut.ssv'
FILE_NAME_EXEC = 'credit_risk_kaggle.csv'
output_file = dir + '/results/output_scores.csv'
logging_file_exec = dir+'/execution_log.log'
decision_tree=True #If there is a decision tree dot file to parse set this variable
dot_file = RESULTS_OUTPUT_DIR+'Decision_Tree.dot'
model_pickle_file=MODELS_OUTPUT_DIR + 'Decision_Tree.pkl'

cutoff=0.146139622 
