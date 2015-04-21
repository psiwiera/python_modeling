# Script to define a bunch of useful functions for the model_execution script

# import libraries
from sklearn.externals import joblib
import data_load
import data_preprocessing_functions as ppfuncs 
import pandas as pd
import utils

def loadDataToScore(input_table, input_schema):
    # build the vector to score
    sql = """DROP TABLE IF EXISTS """+input_schema+"""."""+input_table+""";
        CREATE TABLE """+input_schema+"""."""+input_table+""" AS
        ;"""
    data_load.run_sql_string(sql,'Exec')

    # Now read in the data to score
    initial_data = data_load.psqlLoad(input_table, input_schema, columns='*','Exec')
    data = initial_data
    # logger.info('Initial data size')
    # logger.info(data.shape)
    
    return data, initial_data


# This function checks if there are any columns in the model which are not 
# in the data to score.
# This can happen if there are categories that don't exist (eg in the model there was an
# SE1 postcode but this wasn't found in the data to score. 
# We will need to add them these variables, we assume that we add them with zeros as it can only be from a catToBinary
# which converts categories to a 0/1 flag - if the category doesn't exist as a column
# it must be a 0
def addCols(colsToKeep,data):
    extraCols=[]
    for col in colsToKeep:
        if col not in data.columns:
            extraCols.append(col)
            data[col]=0
    return data,extraCols
    
def renameCols(output_df,var_list):
    # columns in Greenplum must be no more than 63 characters
    # it also makes sense to replace some of the special characters
    # loop over the columns and rename them 
    new_cols_dict={}
    for old_col in var_list:
        # include some text replacements here
        new_col=old_col.replace().replace('[','_').replace(',','_').replace(' ','_').replace('(','').replace(')','')
        new_col=new_col[0:62]
        new_cols_dict[old_col]=new_col
        
    output_df.rename(columns=new_cols_dict,inplace=True)
    
    return output_df

def makePrettyTree(dot_file,data,var_results,RESULTS_OUTPUT_DIR):
    #logger.info('--------------------------------- Prettify Decision Tree -----------------------------------')
    var_list = utils.parseDecTree(dot_file,data.columns)
    # logger.info('Variables in the tree')
    # logger.info(pp.pformat(var_list))

    # Add onto the feature selection and write out to CSV file
    # get the unique column list
    var_list = list(set(var_list))
    var_results_tree = pd.merge(var_results,pd.DataFrame({'columns':var_list,'in_tree':[True]*len(var_list)}),how='left',on='columns')
    var_results_tree.to_csv(RESULTS_OUTPUT_DIR + 'feature_selection_tree.csv')
    
    return var_list
    
def applyFeatureSelection(data,MODELS_OUTPUT_DIR):
    #logger.info('--------------------------------- Feature Selection -----------------------------------')
    # unpickle the feature selection results in order to identify which columns should be removed
    # (if feature selection was used
    var_results = joblib.load(MODELS_OUTPUT_DIR + '/feature_selection.pkl') 
    #logger.info(pp.pformat(var_results))

    # NB need to do it in terms of columns to Keep rather than drop because
    # there could be categories in the new data that don't exist in the training data
    d = var_results.loc[var_results['feat_selection']==True]
    colsToKeep = d['columns']

    data,extraCols = addCols(colsToKeep,data)
    # logger.info('Adding the following columns to the vector to score due to missing categories')
    # logger.info(extraCols)

    data = data[colsToKeep]
    # logger.info('data size after feature selection')
    # logger.info(data.shape)
    
    return data,var_results
   
def applyModel(model_file,data,initial_data,RESULTS_OUTPUT_DIR,test_pickle_file):
    # un-pickle the relevant model 
    clf = joblib.load(model_file) 

    # run the model against the uploaded data
    output = clf.predict_proba(ppfuncs.convertToNumpy(data))

    # re-build the output as a dataframe to put back into the database
    col1 = 'prob_class_'+str(clf.classes_[0])
    col2 = 'prob_class_'+str(clf.classes_[1])

    output_df = pd.DataFrame(output,columns=[col1,col2])

    # join the original data
    # logger.info('data size')
    # logger.info(data.shape)
    # logger.info('output size')
    # logger.info(output_df.shape)
    output_df = pd.merge(data,output_df,how='inner',left_index=True,right_index=True)
    # logger.info('final output size')
    # logger.info(output_df.shape)

    output_df = pd.merge(d,output_df,how='inner',left_index=True,right_index=True) 
    
    logger.info('--------------------------------- Get predicted values -----------------------------------')
    # build a CSV file of the roc curve to help find the correct cutoff value
    # Unpickle the testing data
    test_data = joblib.load(test_pickle_file) 
    rocCurveCSV(clf,test_data['X_test'],test_data['y_test'],RESULTS_OUTPUT_DIR)

    return output_df,output, clf
    
def qaFinalDF(output_df,initial_data,output):
    

    print('Data Values OK')
    
    
def applyRules(input_schema,output_table):
    sql = """ALTER TABLE """ + input_schema + """.""" +output_table + """ ADD COLUMN RULE_FLAG INT;
    UPDATE """ + input_schema + """.""" +output_table + """ AS v
    """
    data_load.run_sql_string(sql,'Exec')
    
def rocCurveCSV(clf,X_test,y_test,RESULTS_OUTPUT_DIR):
    # generate the probabilities again
    test_probas_ = clf.predict_proba(X_test)
    fpr,tpr, thresholds = roc_curve(y_test,test_probas_[:,1])
    # Make into a dataframe and write to a csv file
    roc_df = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
    roc_df.to_csv(RESULTS_OUTPUT_DIR + 'Test_ROC_Curve_Data.csv')
    # plot the ROC curve to help determine the threshold
    modFuncs.test_roc_curve(clf,X_test,y_test)