import csv
from sklearn import linear_model as sklm
from sklearn import cross_validation as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition as dec
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# pipeline functions?
# grid searching for estimator parameters
# backward removal of variables
# scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

def all_data_train(data):
	print data.info()
	print data.dtypes
	X = data[0:, :-1]
	y = data[0:,7]
	model = sklm.LogisticRegression()
	model.fit(X,y)
	model.coef_ 
	model.intercept_

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


if __name__ == "__main__":
	print 'Please run this script from the machine_learning_master script'