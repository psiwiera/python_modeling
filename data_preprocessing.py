from sklearn import preprocessing
from sklearn import feature_selection as featsel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging # used to create output log
import datetime
import signal # used for timeout
import pprint as pp # to make log entries nicer to read
import json
import settings

# Local scripts
import data_preprocessing_functions as ppfuncs 

# define a function to run the various required data_preprocessing functions
# note the optional input parameter to determine whether this is to be run
# in execute mode or not. When execute=True then only the processes required
# by the model execution are included (this excludes for example any row-removal functions)
def main(data, execute=False):
    if execute==False:
        print('pre-preprocessing data size')
        print(data.shape)
        #specify details of logging (So logs outputs exactly which function fed info into log)
        logger = logging.getLogger('Master.Data_Preprocessing.main')
    else:
        logger = logging.getLogger('Exec.Data_Preprocessing.main')
    
    #---------Investigating Data Frame----------------------------------------
    if execute==False:
        logger.info('--------------------------------- Data PP - Investigating Variables -----------------------------------')
        logger.info('Summary information of the dataset: %s', pp.pformat(data.info())) 
        data.describe() # this will tell you how many missing values by variable.
        data.info()
        #Create a data frame with class values following initial load. This will be added to post row selection
        # and outlier removal.
        class_cnts_df = pd.DataFrame()
        class_cnts_df['Initial Load'] = data['class'].value_counts()
    
    #--------Filter rows based on knowledge of dataset--------------------------
    if execute==False: 
        logger.info('data size before row selection: %s', data.shape)
        data = ppfuncs.rowSelection(data)
        # Add class values post row selection to class_cnts data frame.
        class_cnts_df['Post Row Sel'] = data['class'].value_counts()
        logger.info('data size after row selection: %s', data.shape)
    
        var_names = {}
        data = ppfuncs.removeOutliers(data, var_names)
        class_cnts_df['Post Outlier Rem'] = data['class'].value_counts()
        logger.info('post outlier removal: %s', data.shape)
        logger.info('Class distribution post outlier removal, %s', pp.pformat(class_cnts_df))
    
        # Produce histograms of remaining variables and rows to investigate distributions post outlier removal.
        hist_data = data[['duration','credit_amount','savings']]
        
        plt.plot()
        hist_data.hist()
        plt.ylabel('Freq', fontsize=6)
        plt.show()
        
        #plt.plot()
        #hist_data.boxplot(by='class')
        #xplt.show()
        
        # Pivoting dataframe to gain insight into structure and distributions
        ##logger.info('Pivot Table: %s', pp.pformat(ppfuncs.pivotTable(data, , , )))
        # check for duplicates using data.duplicated(). returns an array with trues/falses
        ##pp.pformat(ppfuncs.pivotTable(data, , , ))
        #Need to check the class variable balance (how even is it, do we need stratified sampling?)
    
    #---------Check User Is Happy With Variables before continuing------------
    if execute==False: 
        proceed = input('Are you happy to proceed (y or n)?')
        if proceed == 'n':
            raise SystemExit
        elif proceed != 'y':
            proceed2 = input('Please only enter y or n in lower case: ')
            if proceed2 == 'n':
                raise SystemExit
            elif proceed2 != 'y':
                print('Did not enter y or n, quitting.')
                raise SystemExit
   
   
        # Can use regex to replace values in a dataframe
        print('pre-preprocessing data size')
        print(data.shape)
    


    #---------Modifying Variables----------------------------------------------
    logger.info('--------------------------------- Data PP - Modifying Variables -----------------------------------')
    # Converting values of variables based on if statements
    # Class variable should always be called 'class'
    # Converting categorical variables to numerical to use with sklearn (needs 
    # numpy array and only accpet numerical vars)
    # Convert categorical class variable to binary numerical??

    # modifying the class variable returns the data frame minus the class var and the class var as y.
    data = ppfuncs.modifyClassVariable(data)
    data = ppfuncs.replaceValuesVar1(data)
    data = ppfuncs.mapNewVar1(data)


    # if execute==False:
    #    data = ppfuncs.modifyClassVariable(data) Not required here, as class var already in correct format
        
    
    #Specify which categorical variables to convert to binary
    catVars = {'check_status', 'credit_history', 'purpose', 'savings', 'pres_employ_since',
    'pers_status', 'other_debtors', 'property', 'instal_plans', 'housing', 'job', 'telephone',
    'foreign_worker', 'mod_status'}
    data = ppfuncs.catToBinary(data, catVars)
    print('data size after catToBinary')
    print(data.shape)
    
    # Scale variables to standard normal distribution. Particularly important for SVM models 
    # and others which use distance function.
    # Use from sklearn import preprocessing
    # x = preprocessing.scale(X[:, :3]) will scale the first three vars using all rows of data.

    #----------Missing Values---------------------------------------------------
    logger.info('--------------------------------- Data PP - Address Missing Variables -----------------------------------')
    # Manage & Impute missing values (http://pandas.pydata.org/pandas-docs/stable/
    #missing_data.html). 
    # Need to determine what your strategy is for dealing with missing values.
    # When calculating numeric values using variables with missing 
    # values, ensure you are familiar with pandas behaviour (explained in the link above)

    if execute==False:
        
        # Save the mean values out to a csv for use in the model execution
        f=open(settings.MODELS_OUTPUT_DIR+'replacement_values.json','w')
        f.write(json.dumps({}))
        f.close()
    else:
        f=open(settings.MODELS_OUTPUT_DIR+'replacement_values.json','r')
        x=(f.readlines())
        y=json.loads(x[0])
        f.close()
        
    # data.dropna(axis = 0) # remove rows which have na values (axis = 1 will remove columns
    # with na values)
    # data.interpolate(method='barycentric') # this will replace missing values with an 
    # interpolated value for that variable. Uses Scipy, be careful of what order your rows 
    # are in.


    
    #--------Select Variables for Modelling-------------------------------------
    logger.info('--------------------------------- Data PP - Drop Variables Not For Modelling -----------------------------------')
    # Selecting variables to use in the modeling stage. This is done by specifying the vars to 
    # drop, then using the drop function
    # axis=1 specifies dropping columns not rows
    data = ppfuncs.removeFields(data,execute)
    print('data size after remove fields')
    print(data.shape)
    
    # select only certain vars to move forward with (for testing model effectiveness)
    #data = data[[]]

    #---------Subsetting & Sampling--------------------------------------------
    # sampling - different sampling approaches
    # To include initial sampling if dataset is large, what about training, validation and test sets?
    #over sample class to address class imbalance
    #allposclass = data[data['class_variable'] == 1]
    #data.append(allposclass)
    #data.append(allposclass)
    
    # Separate out class variable from dataset
    if execute==False:
        logger.info('--------------------------------- Data PP - Separate Class Variable -----------------------------------')
        #print(data.groupby('class_variable').describe())
        data, y = ppfuncs.sepClass(data)
        logger.info('Check data frame post class var removal: %s', pp.pformat(data.info()))

    #-----Convert to np array for use with Sklearn-----------------------------
    #print('Pandas dataframe size')
    #print(data.describe())
    
    data_np_array = ppfuncs.convertToNumpy(data)
    
    if execute==False:
        y_np_array = ppfuncs.convertToNumpy(y)
        print('initial numpy array dimensions')
        print(data_np_array.shape)
        
    # Create list of variable names to use later in model building var analysis
    var_names = data.columns.values.tolist()
    # Create a dataframe of var names to incorporate results in later scripts.
    var_results=pd.DataFrame({'columns':var_names})
    # add in here pretty print code.
    logger.info('Variable names list following preprocessing. %s', pp.pformat(var_names))

    if execute==False:
        #---Write Modelling Dataset back to csv/database----
        # writing dataframe/array back to a postgres db
        # create function for people to use
        #np.savetxt(settings.DATA_OUTPUT_DIR + 'data_array.txt', data_np_array)
        #np.savetxt(settings.DATA_OUTPUT_DIR + 'class_array.txt', y_np_array)

        print('data size after pre-processing')
        print(data.shape)
    
    
        # final return of the data preprocessing stage back into the machine learning master.
        return data_np_array, y_np_array, var_results
    else:
        # if we are calling this from the model execution script then return the 
        # data dataframe and the variable list data frame
        return data,var_results

if __name__ == "__main__":
	print ('Please run this script from the machine_learning_master script')
