from sklearn import preprocessing
from sklearn import cross_validation as cv
import numpy as np
import pandas as pd
import logging # used to create a results log.
import datetime
#import matplotlib.pyplot as plt
import pprint as pp # to make log entries nicer to read

# Pandas Data Frame - These functions assume the dataset being used is a Pandas
# data frame.

#-------- Investigate Data Frame Overall ---------------------------
# Create pivot table
def pivotTable(data, indexVar, columnsVar, valuesVar):
	pvtTable = data.pivot_table(index=indexVar, columns=columnsVar, values=valuesVar, aggfunc='count')
	return pvtTable

#-------- Investigate Variable(s)-----------------------------------
# for var in vars....
#add second field to function which is optional and specifies a list of vars to do it on.
# if blank it does it for all vars.


#-------- Modifying Variable Functions -----------------------------
def modifyClassVariable(data):
	# specify what corrections need to take place to the class variable to
	# ensure it conforms with scikit-learn expected formats.
	data['SeriousDlqin2yrs'][data['SeriousDlqin2yrs'] == 1] = 0
	data['SeriousDlqin2yrs'][data['SeriousDlqin2yrs'] == 2] = 1

	return data

def sepClass(data):
	# Separate out class variable from rest of dataset and drop it from that dataset.
	# This is required by scikit-learn which handles the class separately.
	# Should be used as a final data preprocessing step before converting to numpy array. 
	y = data['SeriousDlqin2yrs']
	data = data.drop('SeriousDlqin2yrs', axis=1)

	return data, y

def removeOutliers(data, var_names):
    #Identify and remove rows with outliers for the variables named in var_names
    outlier_dict = {}
    #calculate the outlier values for each variable (do not delete outliers within this iterator)
    for var in var_names:
        q1 = data[var].quantile(0.25)
        q3 = data[var].quantile(0.75)
        mean_val = data[var].mean()
        outlier = mean_val + (3*(q3-q1))
        outlier_dict[var]=outlier
    #delete outliers using dictionary of key value outlier pairs
    for key, val in outlier_dict.items():
        data = data[data[key]<val]
  
    return data  

# At least one function per variable (might require more than one function
# for a variable if multiple manipulations are being executed)

# Turn categorical variable into separate binary variables.
def catToBinary(data, vars_list):
	# Turns the vars listed in vars_list (within data dataframe) into binary vars
	# can use DictVectorizer in sklearn but this uses Pandas functionality
	# This function does not remove the original categorical vars from the df. 
	# Use removeFields function below for this
	for var in vars_list:
		temp = pd.get_dummies(data[var], prefix=var, prefix_sep='=', dummy_na=True)
		data = data.join(temp)

	return data

def replaceValuesVar1(data):
#!!!!!!!!!!------    ---------!!!!!!!!!!!
#Not present in the new data set, so removed this line
#	data['purpose'].replace('A40','Car')
	
	return data

# create new variable mapping an existing variable using lookup table
def mapNewVar1(data):
#!!!!!!!!!!------    ---------!!!!!!!!!!!
#Again, these values are not present in the new data set
	# Similar to an Excel lookup, this converts the values in a variable to new categorical values
#	mapDict = {
#		'A11':'No Savings',
#		'A12':'No Savings',
#		'A13':'Savings',
#		'A14':'Unknown'
#	}
#	data['mod_status'] = data['check_status'].map(mapDict)
	
	return data

    
    #--------- Variable binning for scorecard variables---------------
# consider using the gridsearch optimiser to determine best bins. check binning in scipy and numpy as well.

#--------- Modify overall dataset ---------------------------------
def removeFields(data,execute=False):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    if execute==False:
        logger = logging.getLogger('Master.Data_Preprocessing.ppfuncs.removeFields')
    else:
        logger = logging.getLogger('Exec.Data_Preprocessing.ppfuncs.removeFields')
    # This function drops columns from the data frame. Any variable which is not numerical will need to be
    # dropped. Note this DROPS variables, it does NOT specify which ones to retain in the dataframe.
    #!!!!!!!!!!------    ---------!!!!!!!!!!!
    #Don't want to drop any of these columns, don't exist in new data set
    #colsToDrop = ['check_status', 'credit_history', 'purpose', 'savings', 'pres_employ_since',
	#'pers_status', 'other_debtors', 'property', 'instal_plans', 'housing', 'job', 'telephone',
	#'foreign_worker', 'mod_status']
    colsToDrop = []
    
    logger.info('These variables are dropped before the modelling stage: %s', pp.pformat(colsToDrop))
    
    return data.drop(colsToDrop, axis = 1)

def rowSelection(data):
    
    return data

#--------- Convert to numpy array ---------------------------------
def convertToNumpy(data):
	return data.as_matrix()