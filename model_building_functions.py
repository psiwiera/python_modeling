from sklearn import linear_model as sklm
from sklearn import tree as sktree
from sklearn import cross_validation as cv
from sklearn import decomposition as dec
from sklearn import feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.externals.six import StringIO 
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#import psycopg2
import pandas
from io import StringIO
<<<<<<< HEAD
#from graphviz import Graph 
import pydot 
from inspect import getmembers

#----------Training and Test Set Preparation------------------------------

def basicTrainTest(np_data, y):
	# Create basic training test split without stratification.
	# Does not do stratuified sampling so probably shouldn't use for imbalanced class problems.
	# set the test size to be the proportion of the dataset you want to hold back from training.
	# Cross validation functionality is available in the data modeling script set.
	X_train, X_test, y_train, y_test = cv.train_test_split(np_data, y, test_size=0.20, random_state=33)

	return X_train, X_test, y_train, y_test


def featureSelection(X_train, y_train):
	# Downselect variables based upon feature selection.
	# When training using a large number of variables can use this for an iniiial downselection
	# which will help improve model building performance. percentile selects the vars within the top x
	# percentile based on an evaluation metric.
	featsel = fs.SelectPercentile(fs.chi2, percentile=20)
	X_train_fs = featsel.fit_transform(X_train, y_train)

	return X_train_fs

#----------Model functions-------------------------------------------------
def logReg (model, X,y):
	model = model.fit(X,y)
	print ('Logistic Regression Model Results --------------------------')
	print ('Model coefficients: ')
	print (model.coef_)
	print ('Model intercept: ')
	print (model.intercept_)
=======
>>>>>>> origin/master

	return model

def decTree(model, X, y):
	# cannot cope with missing values so make sure they have been addressed before hand.
	model = model.fit(X,y)
	print ('Decision Tree Model Results --------------------------------')
	print ('Feature importances: ')
	print (model.feature_importances_)

<<<<<<< HEAD
	print(getmembers(model.tree_))

	return model

#---------Model Functions Recursive Feature Selection with Cross Validation-
def rfeCv(modEstimator,X,y):
	# uses the rfecv functionality to identify the optimum number of variables (features)
	# along with cross validation to determine the most optimum model.
	rfecv = fs.RFECV(estimator=modEstimator, step=1, cv=3, 
		scoring='accuracy', verbose=10)
	rfecv.fit(X, y)

	print("Optimal number of features : %d" % rfecv.n_features_)

	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()

	return rfecv

=======
# Model functions
def logReg (X,y):
	model = sklm.LogisticRegression()
	model.fit(X,y)
	print('Logistic Regression Model Results --------------------------')
	print('Model coefficients: ')
	print(model.coef_)
	print('Model intercept: ')
	print(model.intercept_)

	return model

def decTree(X,y):
	model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
		min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, 
		min_density=None, compute_importances=None, max_leaf_nodes=None)
	model.fit(X,y)
	print('Decision Tree Model Results --------------------------------')
	print('Feature importances: ')
	print(model.feature_importances_)
	print('Tree')
	#tree = StringIO()
	#tree = tree.export_graphviz(model, out_file='tree.dot')
	# sprint tree.getvalue()
>>>>>>> origin/master

# Use cross validation to build model.
def crossValScore(unbuilt_model,X,y):
	cross_val_score(unbuilt_model, X, y, cv=10) 

#---------- Visualisation Functions---------------------------------------
def plotDecTree(tree):
	# saves the model in dot format to visualise using GraphViz
	
	dot_data = StringIO() 
	sktree.export_graphviz(tree, out_file='dot_data.dot') 
	#decTreeGraph = pydot.graph_from_dot_data(dot_data.getvalue())
	#decTreeGraph.write_pdf("dectree.pdf")

# Define function to Plot ROC curve
def plotRocCurve(classifiers):
	pl.clf()
	#pl.plot(log_fpr, log_tpr, label='Logistic Regression ROC curve (area = %0.2f)' % log_roc_auc)

	for key,value in classifiers.items():
		pl.plot(value['false_positive'], value['true_positive'], label=str(value['label']) + ' (area = %0.2f)' % value['ROC_auc'])
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic example')
	pl.legend(loc="lower right")
	pl.show()


def pipeline(data):
	clf = Pipeline([
		('feature_selection', LinearSVC(penalty="l1")), 
		('classification', RandomForestClassifier())])
	clf.fit(X, y)
	
