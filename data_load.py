import csv
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


def csvfile(INPUT_DIR, file_name, RESULTS_OUTPUT_DIR):
	#---------Load Data-----------------------
	#load csv into pandas data frame using pandas read_csv function
	try:
		data = pd.read_csv(INPUT_DIR + file_name,
			sep = ' ', #using this to set what separator is used (e.g. , or | etc)
			header = 0) #this tells the parser if there is a header and on which row
	except IOError:
		print 'cannot open file'
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
	#data['check_status'].hist(by=data['duration'])
	#plt.savefig(RESULTS_OUTPUT_DIR + 'test2.png')
	print 'Data Load Completed Successfully'
	return data

def postgres(tbd):
	return tbd

if __name__ == "__main__":
	print 'Please run this script from the machine_learning_master script'
