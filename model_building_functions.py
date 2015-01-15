from sklearn import linear_model as sklm
from sklearn import tree
from sklearn import cross_validation as cv
from sklearn import decomposition as dec
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#import psycopg2
import pandas
from io import StringIO


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

# Use cross validation to build model.
def crossValScore(X,y, unbuilt_model):
	cross_val_score(unbuilt_model, X, y, cv=10) 

#---------- Visualisation Functions
def plotDecTree(tree, RESULTS_DIR):
	# saves the model in dot format to visualise using GraphViz
	tree = tree.export_graphviz(model, out_file=RESULTS_DIR + 'tree.dot')
	print.tree

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
	X_train, X_test, y_train, y_test = cv.train_test_split(data[0:, :-1], data[0:,7])
	model = sklm.LogisticRegression()
	model.fit(X_train,y_train)
	model.score(X_test, y_test)

	logistic = sklm.LogisticRegression()

	pca = dec.PCA()
	pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

	###############################################################################
	# Plot the PCA spectrum
	pca.fit(X_train)

	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')

	###############################################################################
	# Prediction

	n_components = [20, 40, 64]
	Cs = np.logspace(-4, 4, 3)

	estimator = GridSearchCV(pipe,
	                         dict(pca__n_components=n_components,
	                              logistic__C=Cs))
	estimator.fit(X_train, y_train)

	plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
	            linestyle=':', label='n_components chosen')
	plt.legend(prop=dict(size=12))