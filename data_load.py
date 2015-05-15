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
# import psycopg2
import glob
from sqlalchemy import create_engine
import datetime
from sqlalchemy.types import *
import codecs
import os
import logging
import datetime
import pprint as pp # to make log entries nicer to read


def csvfile(INPUT_DIR, file_name, RESULTS_OUTPUT_DIR):
    #specify details of logging (So logs outputs exactly which function fed info into log
    logger = logging.getLogger('Master.Data_load.csvfile')
    #---------Load Data-----------------------
    #load csv into pandas data frame using pandas read_csv function
    #!!!!!!!!!!------  separator changes from ' ' to ','  ---------!!!!!!!!!!!
    
    try:
        data = pd.read_csv(INPUT_DIR + file_name,
                           sep = ',', #using this to set what separator is used (e.g. , or | etc)
                           header = 0) #this tells the parser if there is a header and on which row
        # alternatively reading data from a sql db
        # create function/separate python script for people to use.

        #----------Check Initial Data Load----------
        # Check the dataset is as expected and the data types have been interpreted correctly.
        # check data type for each data type
        logger.info('Initial Data Info')
        logger.info(data.dtypes)
        # Look at first few rows of data to check read in correctly
        logger.info(data.head())
        # look at summary statistics of dataset to check if as expected
        logger.info(data.describe())
        logger.info(data.info())

        # More detailed analysis of specific variables
        # plot histogram of a variable grouped by a categorical var
        #data['check_status'].hist(by=data['duration'])
        #plt.savefig(RESULTS_OUTPUT_DIR + 'test2.png')
        logger.info ('Data Load Completed Successfully')
        print('Data Load Completed Successfully')
        
        return data
        
        
    except IOError:
        logger.info ('cannot open file %s from folder %s', file_name, INPUT_DIR)



# setup a connection
# function to setup a pg connection
# import psycopg2
def pgconnect(logtype='Master'):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger(logtype+'.Data_load.pgconnect')
    try:
        # Set up a database connection. 
        # include in try..catch in case the connection fails
        conn = psycopg2.connect()
        return conn
    except Exception:
        # This is pretty ugly... probably a better way
        # using the sys.exit() means you don't get the 
        # full traceback of the error. This is fine for ipython 
        # but exits python from a shell command - which is why its
        # removed
        # sys.exit('Cannot open database connection')
        logger.info('Cannot open database connection')
        

# define function to read in data for the avazu kaggle
#df1 = psqlLoad('training_sample','kaggle_avazu',columns='id,click')
#df2 = psqlLoad('training_sample','kaggle_avazu',columns='*')
#df3 = psqlLoad('training_sample','kaggle_avazu',columns=['id','click'])
#df4 = psqlLoad('training_sample','kaggle_avazu')
def psqlLoad(table, schema, columns='*',logtype='Master'):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger(logtype+'.Data_load.psqlLoad')
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
    conn = pgconnect(logtype)
    
    # build the query - as I say currently this is vulnerable to SQL injection
    # but I don't think we need to worry - anyone who has the priveleges needed
    # to run this command would also have the privileges to do what they like
    # to the database anyway...
    sql = "SELECT " + columns_string + " FROM " + schema + "." + table + ";"
    
    # Read data from the SQL command into a dataframe
    
    logger.info('Loading vector from database %s', datetime.datetime.now().time().isoformat())
    df = pd.read_sql(sql, conn)
    logger.info('Data load from database complete %s', datetime.datetime.now().time().isoformat())
    print('Data load from database complete', datetime.datetime.now().time().isoformat())
    # return the dataframe
    return df


# define a function to write a pandas dataframe back to the database
# Note this function will fail if the dataframe contains array objects
def psqlWrite(df, table_name, schema_name,logtype='Master'):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger(logtype+'.Data_load.psqlWrite')
    
    engine = create_engine()
    # Need to include some error handling to improve this a bit...
    df.to_sql(table_name, engine, schema=schema_name, if_exists='fail',index_label='pd_index') 

    #  The basic python method to upload is really slow. So use the bulk
    # uploader instead, this basically writes out a csv file and then loads 
    # that as an external table
   
def psqlBulkWrite(df,table_name,schema_name,logtype='Master'):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger(logtype+'.Data_load.psqlBulkWrite')
    # First write out the csv
    columns_list = list(df.columns.values)
    df.to_csv('temp/python_load.csv', index_label='pd_index',columns=columns_list, sep="|")

    # setup a greenplum connection
    conn = pgconnect(logtype)
    db=conn.cursor()

    # create the new table with the correct columns
    # it would be nice to use an empty dataframe to do this, but that doesn't upload the data
    # types particularly well. Instead create a one-row data frame and then truncate the table??
    #df_dummy = df.drop(df.index[1:df.shape[0]])
    #psqlWrite(df_dummy,table_name,schema_name)
    #db.execute('truncate table '+schema_name+'.'+table_name)

    # Alternative method - write an empty table but generate a datatypes column
    # using sqlalchemy types (which to_sql understands)
    # create empty dataframe
    df_dummy = df.drop(df.index)
    # create dictionary mapping pandas (numpy) datatypes to sqlalchemy types
    pd_type_map = {np.dtype('int8'):INT,np.dtype('int16'):INT,np.dtype('int32'):INT,np.dtype('int64'):BIGINT,np.dtype('float16'):NUMERIC,np.dtype('float32'):NUMERIC,np.dtype('float64'):NUMERIC,np.dtype('bool'):BOOLEAN,np.dtype('datetime64[ns]'):TIMESTAMP,np.dtype('object'):TEXT}

    # use the dictionary mapping to get the sqlalchemy types for each column
    dtypes_list=[pd_type_map[i] for i in df_dummy.dtypes.values]

    # build a dictionary of column name and sqlalchemy data type
    dtypes_dict = dict(zip(df_dummy.columns,dtypes_list))
    # write the empty dataframe as a table
    engine = create_engine()
    df_dummy.to_sql(table_name, engine, schema=schema_name, if_exists='fail',index_label='pd_index',dtype=dtypes_dict)

    # now use the bulk upload method
    # drop the external table if it already exists
    db.execute('drop external table if exists ' + schema_name + '.python_load;')

    # create the external table in the write schema
    db.execute("create external table "+schema_name+".python_load (LIKE "+schema_name+"."+table_name+") LOCATION ('python_load.csv') FORMAT 'CSV' (HEADER DELIMITER AS '|' NULL '');")

    # Insert the rows from the external table into the correct table
    db.execute('INSERT INTO '+schema_name+'.'+table_name+' SELECT * FROM '+schema_name+'.python_load;')

    # Drop the external table
    #db.execute('drop external table if exists ' + schema_name + '.python_load;')

    # commit the commands to the database
    conn.commit()
    
    logger.info('Bulk write committed successful')
    # close the database connection
    db.close()
    conn.close()

    # delete the intermediate csv file

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
def run_sql_scripts(directory,logtype='Master'):
    # setup a pg connection
    conn = pgconnect(logtype)
    db=conn.cursor()
    
    scripts = get_dir_sql_scripts(directory)
    numScripts = len(scripts)
    
    for i,sqlFile in enumerate(scripts):
        print('Running script '+str(i+1)+' of '+str(numScripts))
        db.execute(open(sqlFile, "r").read())
    
def read_pg_script(file):
	# read the bytes in the file to test if there is a byte-order-mark
	bytes=min(32,os.path.getsize(file))
	raw=open(file,'rb').read(bytes)
	
	# test byte-order-mark
	if raw.startswith(codecs.BOM_UTF8):
		infile = open(file, 'r',encoding='utf-8-sig')
	else:
		infile = open(file, 'r')
		#result=chardet.detect(raw)
		#encoding=result['encoding']
		
	# readfile and return the contents
	data = infile.read()
	infile.close()
	
	return data
	
	
# define function to run single script
def run_single_sql_script(file,logtype='Master'):
	# setup a pg connection
	conn = pgconnect(logtype)
	db=conn.cursor()
	# read the file and execute the code
	db.execute(read_pg_script(file))
	# commit and close
	conn.commit()
	db.close()
	conn.close()

def run_sql_string(sqlString,logtype='Master'):
	conn = pgconnect(logtype)
	db=conn.cursor()
	db.execute(sqlString)
	conn.commit()
	db.close()
	conn.close()
	
	return ('Completed')

#Function to read data from pivotal HD node - what format will this be?? SQL?
def read_from_pivotal():
	pass

if __name__ == "__main__":
	print ('Please run this script from the machine_learning_master script')
