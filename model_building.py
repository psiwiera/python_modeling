import csv
from sklearn import linear_model as sklm
from sklearn import tree as sktree
from sklearn import cross_validation as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition as dec
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

import model_building_functions as modfuncs 

# pipeline functions?
# grid searching for estimator parameters
# backward removal of variables
# scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

def modelsBuild(np_data, y):
	# no_data is whole data set minus the class variable which is y.

	#-----Create Training and Test sets----------------------------------------
	X_train, X_test, y_train, y_test = modfuncs.basicTrainTest(np_data, y)
	
	#-----Variable Selection (Feature Selection)-------------------------------
	# Use scikit-learn feature selection algorithms to downselect variables.
	# data must be as a np_array 
	# Add feature selection preprocessing step here (not RFE CV)


	#----- Model Building and Comparison---------------------------------------
	# Basic Models
	# Decision Tree
	modelDecTree1 = sktree.DecisionTreeClassifier()
	modelDecTree1 = modfuncs.decTree(modelDecTree1, X_train, y_train)
	#modfuncs.plotDecTree(modelDecTree)

	# Logistic Regression
	logreg1 = sklm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
	modelLogReg = modfuncs.logReg(logreg1, X_train, y_train)
	

	# Recursively build model to determine features
	logreg2 = sklm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
	modelRfeLogReg2 = modfuncs.rfeCv(logreg2, X_train, y_train)


if __name__ == "__main__":
	print ('Please run this script from the machine_learning_master script')
