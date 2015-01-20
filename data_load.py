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
		print('cannot open file')
	# alternatively reading data from a sql db
	# create function/separate python script for people to use.

	#----------Check Initial Data Load----------
	# Check the dataset is as expected and the data types have been interpreted correctly.
	# check data type for each data type
	print(data.dtypes)
	# Look at first few rows of data to check read in correctly
	print(data.head())
	# look at summary statistics of dataset to check if as expected
	print(data.describe())
	print(data.info())

	# More detailed analysis of specific variables
	# plot histogram of a variable grouped by a categorical var
	#data['check_status'].hist(by=data['duration'])
	#plt.savefig(RESULTS_OUTPUT_DIR + 'test2.png')
	print('Data Load Completed Successfully')
	return data

# define function to read in data for the avazu kaggle
#df1 = psqlLoad('training_sample','kaggle_avazu',columns='id,click')
#df2 = psqlLoad('training_sample','kaggle_avazu',columns='*')
#df3 = psqlLoad('training_sample','kaggle_avazu',columns=['id','click'])
#df4 = psqlLoad('training_sample','kaggle_avazu')
def psqlLoad(table, schema, columns='*'):
    # Use py-postgresql because it handles the authentication stuff
    #import pandas
    #import postgresql
    #import numpy
    # May need to think about SQL injection
    
    # Use a try catch to test whether we can get a connection
    # No doubt we could make this a bit cleaner
    try:
        # Set up a database connection. 
        # Note exchange localdb with hodacstag and localhost with dasgpdb01 
        # include in try..catch in case the connection fails
        db = postgresql.open("pq://localhost/localdb")
    except Exception:
        print('Cannot open database connection')
    else:
        # ought to include some testing of whether the columns/table/schema exist
        # handle different column input types
        # if the default or * is used
        if columns=='*':
            columns_ps=db.prepare("select column_name from information_schema.columns where table_name='" + table + "' and table_schema='" + schema + "'")
            columns=numpy.array(columns_ps())
            columns_array = columns.flatten()
            columns_string = ",".join(columns_array)
        # if a simple string of comma separated columns is used
        elif type(columns) is str:
            columns_string = columns
            columns_array = columns.split(',')
        else:
            columns_array = columns
            columns_string = ",".join(columns)
            
        # prepare the statement
        data_ps = db.prepare("select " + columns_string + " from " + schema + "." + table )
        # data_ps = db.prepare("select * from " + schema + "." + table )
        
        # Read data from the SQL command into a numpy array and convert to a dataframe
        data=numpy.array(data_ps())
        df = pandas.DataFrame(data)
        
        # add the columns to the dataframe
        df.columns = columns_array
        
        # return the dataframe
        return df


# Define a function to run all SQL files found in a given directory, run in order of name
def run_dir_sql_scripts(directory):
	"Function which provides as its output a list of SQL scripts located in the input directory "
	file_list = glob.glob(directory + '*.sql')
	file_list.sort()
	run_sql_scripts(file_list)
    
# Define a function to run a SQL script(s) 
# example: /Users/mthomson/sql/kaggle/avazu/variable_1.sql
# The input is a list of files to run
# This function currently runs using psql, if we get py-postgresql working 
# then I wonder if we can use that.....
def run_sql_scripts(file_list):
	"Function runs a number of SQL scripts in a single transaction using PSQL. Settings may need to be set within the code at the moment"
	# Note that they need to be in the correct order
	# prepend the cat command
	file_list.insert(0,'cat')
	# In order to only require the user to enter their password once we need to cat all the files together
	# and pass to psql as one transaction
	ps1 = subprocess.Popen(file_list, stdout=subprocess.PIPE)
	ps2 = subprocess.Popen(("/Applications/pgAdmin3.app/Contents/SharedSupport/psql", "--host=localhost", "--port=5432", "--dbname=localdb", "-1", "-f", "-"), stdin=ps1.stdout)
	# the wait command means that it waits for the user to enter their password before continuing
	ps2.wait()



if __name__ == "__main__":
	print('Please run this script from the machine_learning_master script')
