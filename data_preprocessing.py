from sklearn import preprocessing
from sklearn import feature_selection as featsel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local scripts
import data_preprocessing_functions as ppfuncs 

def main(data, DATA_OUTPUT_DIR):
	
<<<<<<< HEAD
	#---------Investigating Data Frame----------------------------------------
	print ('Summary information of the dataset: ')  
	print (data.info()) # this will tell 
=======
	print('Summary information of the dataset: ')
	print(data.info()) # this will tell 
>>>>>>> origin/master
	# you how many missing values by variable.
	# Can use regex to replace values in a dataframe

	# Pivoting dataframe to gain insight into structure and distributions
	print (ppfuncs.pivotTable(data, 'purpose', 'pres_employ_since', 'savings'))
	# check for duplicates using data.duplicated(). returns an array with trues/falses


	#---------Modifying Variables----------------------------------------------
	# Converting values of variables based on if statements
	# Class variable should always be called 'class'
	# Converting categorical variables to numerical to use with sklearn (needs 
	# numpy array and only accpet numerical vars)
	# Convert categorical class variable to binary numerical
	# modifying the class variable returns the data frame minus the class var and the class var as y.
	data = ppfuncs.modifyClassVariable(data)
	data = ppfuncs.replaceValuesVar1(data)
	data = ppfuncs.mapNewVar1(data)
	# Specify which categorical variables to convert to binary
	catVars = {'check_status', 'credit_history', 'purpose', 'savings', 'pres_employ_since',
	'pers_status', 'other_debtors', 'property', 'instal_plans', 'housing', 'job', 'telephone',
	'foreign_worker', 'mod_status'}
	data = ppfuncs.catToBinary(data, catVars)
	print (data.info())

	print (data['purpose'])

	# Scale variables to standard normal distribution. Particularly important for SVM models 
	# and others which use distance function.
	# Use from sklearn import preprocessing
	# x = preprocessing.scale(X[:, :3]) will scale the firs three vars using all rows of data.

	#----------Missing Values---------------------------------------------------
	# Manage & Impute missing values (http://pandas.pydata.org/pandas-docs/stable/
	#missing_data.html). 
	# Need to determine what your strategy is for dealing with missing values.
	# When calculating numeric values using variables with missing 
	# values, ensure you are familiar with pandas behaviour (explained in the link above)
	# data['var_name'].fillna(value)
	# data.dropna(axis = 0) # remove rows which have na values (axis = 1 will remove columns
	# with na values)
	# data.interpolate(method='barycentric') # this will replace missing values with an 
	# interpolated value for that variable. Uses Scipy, be careful of what order your rows 
	# are in.

	#--------Select Variables for Modeling-------------------------------------
	# Selecting variables to use in the modeling stage. This is done by specifying the vars to 
	# drop, then using the drop function
	# axis=1 specifies dropping columns not rows

	data = ppfuncs.removeFields(data)
	
	#---------Subsetting & Sampling--------------------------------------------
	# sampling - different sampling approaches
	# To include initial sampling if dataset is large, what about training, validation and test sets?

	# Separate out class variable from dataset
	data, y = ppfuncs.sepClass(data)

	#-----Convert to np array for use with Sklearn-----------------------------
	data_np_array = ppfuncs.convertToNumpy(data)
	y_np_array = ppfuncs.convertToNumpy(y)

	#---Write Modeling Dataset back to csv/database----
	# writing dataframe/array back to a postgres db
	# create function for people to use
	np.savetxt(DATA_OUTPUT_DIR + 'data_array.txt', data_np_array)
	np.savetxt(DATA_OUTPUT_DIR + 'class_array.txt', y_np_array)
	
	# final return of the data preprocessing stage back into the machine learning master.
	return data_np_array, y_np_array

if __name__ == "__main__":
<<<<<<< HEAD
	print ('Please run this script from the machine_learning_master script')
=======
	print('Please run this script from the machine_learning_master script')
>>>>>>> origin/master
