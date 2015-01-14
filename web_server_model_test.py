# This script sets up a webserver with the python microframework Flask
# this is primarily for demo purposes only with a more permanent solution required eventually

# import relevant modules
from flask import Flask
import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from flask import send_from_directory
import csv
from sklearn.externals import joblib
from sklearn import svm, tree, datasets
import numpy
from numpy import genfromtxt
import pandas
import json

# to begin with the system is called by uploading a file containing a json object
# In order to upload a file we need to set a folder location on the webserver (in this case on my laptop)
UPLOAD_FOLDER = '/Users/mthomson/web_dev/upload_files/'
# we can use some error handling to ensure we only get the right file types
# for this test restrict to json
ALLOWED_EXTENSIONS = set(['json'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# extract the model from a pre-pickled version
# eventually we probably want this to be a dictionary of all our models where the 
# model to use is included as a separate parameter on upload
clf = joblib.load('/Users/mthomson/web_dev/pickled_models/decision_tree_test.pkl') 


# Define a function to upload and read a file containing a json object
# Once server running use curl command:
#  curl -i -F filedata=@kaggle_avazu_test_json.json 127.0.0.1:5000/file_upload
# to upload a file
@app.route('/file_upload', methods=['POST','GET','PUT'])
def upload_file():
	if request.method in ['POST','PUT']:
	    file = request.files['filedata']
	    if file and allowed_file(file.filename):
	        filename=secure_filename(file.filename)
	        #print(filename)

	    return predict(app.config['UPLOAD_FOLDER'] + filename)
	else:
		return '''
    	<!doctype html>
    	<title>Upload new File</title>
    	<h1>Upload new File</h1>
    	<form action="" method=POST enctype=multipart/form-data>
      		<p><input type=file name=filedata>
       		   <input type=submit value=Upload>
    	</form>
    	'''
# call with curl using:
# curl -H "Content-Type: application/json" -X POST -d '[{"0":1005,"1":15707,"2":320,"3":50,"4":1722,"5":0,"6":35,"7":100084,"8":79,"9":16712}]' 127.0.0.1:5000/json_input
@app.route('/json_input', methods=['POST','PUT','GET'])
def json_input():
	if request.method in ['POST','PUT']:
		#print(request.get_json())
		return predict(request.get_json())#predict(request.form['json'])
	else:
		return '''
    	<!doctype html>
    	<title>Please post JSON object</title>
    	<h1>Please post JSON object</h1>
    	'''

# Define a function to check the file type is a json object
def allowed_file(filename):
     return '.' in filename and \
            filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Function to run a predictive model that has been pre-defined, pickled and loaded on starting the webserver
# and apply the predictive model to the input dataset
def predict(input_data):
	# read in the data from the json object (this can be either a file or string) and convert
	# to a numpy array
	#print(input_data)
	df=pandas.io.json.read_json(json.dumps(input_data))
	X_test = df.as_matrix()

	# now run the model against the uploaded data
	output = clf.predict_proba(X_test)

	# return the output. For the time being this is a json with the output predictions
	# - it is up to the calling program to do something useful
	return json.dumps(output.tolist())
	#return json.dumps(input_data)


if __name__ == "__main__":
    app.run(debug=True)