import csv
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

def main(data, DATA_OUTPUT_DIR):
	#----------Missing Values-------------------
	# Manage & Impute missing values (http://pandas.pydata.org/pandas-docs/stable/missing_data.html)
	# When calculating numeric values using variables with missing values, ensure you are familiar with pandas behaviour 
	# (explained in the link above)
	print data.info() # this will tell you how many missing values by variable.
	# data['var_name'].fillna(value) # replace na in a specific var (var_name) with a specific value (value can be a string)
	# data.dropna(axis = 0) # remove rows which have na values (axis = 1 will remove columns with na values)
	# data.interpolate() # this will replace missing values with an interpolated value for that variable. Careful of what order your rows are in.
	# scipy can be used for more complicated interpolation (e.g. data.interpolate(method='barycentric'))
	# Can use regex to replace values in a dataframe

	#---------Modifying Variables---------------
	# Converting values of variables based on if statements

	# Converting categorical variables to numerical to use with sklearn (needs numpy array and only accpet numerical vars)
	# Convert categorical class variable to binary numerical
	data['class'][data['class'] == 1] = 0
	data['class'][data['class'] == 2] = 1

	#--------Select Variables for Modeling-----
	# Selecting variables to use in the modeling stage
	# This is done by specifying the vars to drop, then using the drop function
	colsToDrop = ['check_status', 'credit_history', 'purpose', 'savings', 'pres_employ_since',
	'pers_status', 'other_debtors', 'property', 'instal_plans', 'housing', 'job', 'telephone',
	'foreign_worker']
	data = data.drop(colsToDrop, axis = 1) # axis=1 specifies dropping columns not rows
	print data.dtypes
	print data.describe()

	#---------Sampling----------------------------
	# sampling - different sampling approaches
	# To include initial sampling if dataset is large, what about training, validation and test sets?

	#-----Convert to Dict for use with Sklearn----
	data_np_array = data.as_matrix() 
	print data_np_array

	#---Write Modeling Dataset back to csv/database----
	# writing dataframe/array back to a postgres db
	# create function for people to use
	np.savetxt(DATA_OUTPUT_DIR + 'data_array.txt', data_np_array)
	return data_np_array

if __name__ == "__main__":
	print 'Please run this script from the machine_learning_master script'
