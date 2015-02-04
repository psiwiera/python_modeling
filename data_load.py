import csv
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import sys
import psycopg2
import glob


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

# setup a connection
# function to setup a pg connection
# import psycopg2
def pgconnect():
    try:
        # Set up a database connection. 
        # Note exchange localdb with hodacstag and localhost with dasgpdb01 
        # include in try..catch in case the connection fails
        conn = psycopg2.connect("host=localhost dbname=localdb")
        return conn
    except Exception:
        # This is pretty ugly... probably a better way
        # using the sys.exit() means you don't get the 
        # full traceback of the error. This is fine for ipython 
        # but exits python from a shell command - which is why its
        # removed
        # sys.exit('Cannot open database connection')
        print('Cannot open database connection')
        

# define function to read in data for the avazu kaggle
#df1 = psqlLoad('training_sample','kaggle_avazu',columns='id,click')
#df2 = psqlLoad('training_sample','kaggle_avazu',columns='*')
#df3 = psqlLoad('training_sample','kaggle_avazu',columns=['id','click'])
#df4 = psqlLoad('training_sample','kaggle_avazu')
def psqlLoad(table, schema, columns='*'):
    #import pandas
    #import psycopg2
    # May need to think about SQL injection - don't worry about it for the time being

    # determine the columns to use
    # handle if a list is used
    if type(columns) is list:
        columns_string = ",".join(columns)
    else:
        # otherwise assume the input is a string (or the default)
        columns_string = columns

    # setup a connection
    conn = pgconnect()

    # build the query - as I say currently this is vulnerable to SQL injection
    sql = "SELECT " + columns_string + " FROM " + schema + "." + table + ";"
    
    # Read data from the SQL command into a dataframe
    df = pd.read_sql(sql, conn)

    # return the dataframe
    return df


# Define a function to find all SQL files found in a given directory, sorted in order of name
# note currently there is no error handling for if you enter an empty directory or anything like that
def get_dir_sql_scripts(directory):
	"Function which provides as its output a list of SQL scripts located in the input directory "
	file_list = glob.glob(directory + '/*.sql')
	file_list.sort()
	return file_list
    
# Define a function to run SQL script(s) in a directory
# example: /Users/mthomson/sql/kaggle/avazu/variables/
# The input is a directory containing sql files to run
def run_sql_scripts(directory):
    # setup a pg connection
    conn = pgconnect()
    db=conn.cursor()
    
    scripts = get_dir_sql_scripts(directory)
    numScripts = len(scripts)
    
    for i,sqlFile in enumerate(scripts):
        print('Running script '+str(i+1)+' of '+str(numScripts))
        db.execute(open(sqlFile, "r").read())
    
# define function to run single script
def run_single_sql_script(file):
    # setup a pg connection
    conn = pgconnect()
    db=conn.cursor()
    db.execute(open(file, "r").read())


if __name__ == "__main__":
	print('Please run this script from the machine_learning_master script')
