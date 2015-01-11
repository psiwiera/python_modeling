from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pandas Data Frame - These functions assume the dataset being used is a Pandas
# data frame.

#-------- Investigate Data Frame Overall ---------------------------

#-------- Investigate Variable(s)-----------------------------------
# for var in vars....

#-------- Modifying Variable Functions -----------------------------
def modifyClassVariable(data):
	# specify what corrections need to take place to the class variable to
	# ensure it conforms with scikit-learn expected formats.
	data['class'][data['class'] == 1] = 0
	data['class'][data['class'] == 2] = 1

	return data

# At least one function per variable (might require more than one function
# for a variable if multiple manipulations are being executed)

def modifyVar1(data):
	return data

#--------- Modify overall dataset ---------------------------------
def removeFields(data):
	colsToDrop = ['check_status', 'credit_history', 'purpose', 'savings', 'pres_employ_since',
	'pers_status', 'other_debtors', 'property', 'instal_plans', 'housing', 'job', 'telephone',
	'foreign_worker']
	return data.drop(colsToDrop, axis = 1)

#--------- Subsetting Dataframe -----------------------------------


#--------- Convert to numpy array ---------------------------------
def convertToNumpy(data):
	return data.as_matrix()