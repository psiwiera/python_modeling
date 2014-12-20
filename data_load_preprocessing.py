import csv
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

INPUT_DIR = '/Users/tobybalfre/Documents/Data Science and Analytics/Example Datasets/'
OUTPUT_DIR = '/Users/tobybalfre/Documents/Data Science and Analytics/Example Datasets/'

#---------Load Data-----------------------
#load csv into pandas data frame using pandas read_csv function
data = pd.read_csv(INPUT_DIR + 'german_credit_car_fraud.ssv',
	sep = ' ', #using this to set what separator is used (e.g. , or | etc)
	header = 0) #this tells the parser if there is a header and on which row

# alternatively reading data from a sql db
# create function/separate python script for people to use.

#----------Check Initial Data Load----------
# Check the dataset is as expected and the data types have been interpreted correctly.
# check data type for each data type
print data.dtypes
# Look at first few rows of data to check read in correctly
print data.head()
# look at summary statistics of dataset to check if as expected
print data.describe()
print data.info()

# More detailed analysis of specific variables
# plot histogram of a variable grouped by a categorical var
data['check_status'].hist(by=data['duration'])
plt.savefig(OUTPUT_DIR + 'test2.png')

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

#--------Select Variables for Modeling-----
# Selecting variables to use in the modeling stage
# This is done by specifying the vars to drop, then using the drop function
colsToDrop = ['duration', 'credit_history', 'credit_amount', 'savings',
'instal_rate', 'pers_status', 'resid_since', 'age', 'instal_plans', 'no_exist_credits',
'mainten_no', 'telephone']
data = data.drop(colsToDrop, axis = 1) # axis=1 specifies dropping columns not rows
print data.dtypes
print data.describe()

#---------Sampling----------------------------
# sampling - different sampling approaches
# To include initial sampling if dataset is large, what about training, validation and test sets?


#---Write Modeling Dataset back to csv/database----
# writing dataframe/array back to a postgres db
# create function for people to use 