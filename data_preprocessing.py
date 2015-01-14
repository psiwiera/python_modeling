from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_preprocessing_functions as ppfuncs 

def main(data, DATA_OUTPUT_DIR):
	
	print 'Summary information of the dataset: '  
	print data.info() # this will tell 
	# you how many missing values by variable.
	# Can use regex to replace values in a dataframe

	#---------Modifying Variables----------------------------------------------
	# Converting values of variables based on if statements
	# Class variable should always be called 'class'
	# Converting categorical variables to numerical to use with sklearn (needs 
	# numpy array and only accpet numerical vars)
	# Convert categorical class variable to binary numerical
	data = ppfuncs.modifyClassVariable(data)
	data = ppfuncs.modifyVar1(data)

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
	# Selecting variables to use in the modeling stage. This is done by specifying the vars to drop, then using the drop function
	# axis=1 specifies dropping columns not rows

	data = ppfuncs.removeFields(data)
	
	#---------Subsetting & Sampling--------------------------------------------
	# sampling - different sampling approaches
	# To include initial sampling if dataset is large, what about training, validation and test sets?


	#-----Convert to Dict for use with Sklearn----
	data_np_array = ppfuncs.convertToNumpy(data)

	

	#---Write Modeling Dataset back to csv/database----
	# writing dataframe/array back to a postgres db
	# create function for people to use
	np.savetxt(DATA_OUTPUT_DIR + 'data_array.txt', data_np_array)
	return data_np_array

if __name__ == "__main__":
	print 'Please run this script from the machine_learning_master script'
