import csv
from sklearn import linear_model as sklm
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

def simpleModel(np_data):
	X = np_data[0:, :-1]
	y = np_data[0:,7]

	modelLogReg = modfuncs.logReg(X,y)
	modelDecTree = modfuncs.decTree(X,y)	




if __name__ == "__main__":
	print 'Please run this script from the machine_learning_master script'