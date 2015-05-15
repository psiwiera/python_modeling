# Import libraries
from sklearn import cross_validation as cv
import numpy as np
from sklearn import decomposition as dec
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import feature_selection as fs
import model_building_functions as modfuncs 
import logging
import datetime
import pprint as pp # to make log entries nicer to read
from sklearn.externals import joblib # so that we can output the models
import settings

def modelsBuild(np_data, y, var_results, logger):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build')

    # np_data is whole data set minus the class variable which is y.
    #-----Variable Normalisation/Standardisation-----------------------------------------------
    if settings.variable_scaling==True:
        np_data = modfuncs.scaler(np_data)
    
    #-----Variable Selection (Feature Selection)-------------------------------
    # Use scikit-learn feature selection algorithms to downselect variables.
    if settings.feature_selection==True:
        np_data_fs, var_results = modfuncs.featureSelection(np_data, y, var_results,settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR)
    else:
        np_data_fs = np_data
        
    logger.info('--------------------------------- Modelling - Create Test and Training Sets -----------------------------------')
    
    X_train,y_train,X_test,y_test = modfuncs.kfoldTrainingSet(np_data_fs,y,n_folds=3)
    
    # pickle the test dataset
    joblib.dump({'X_test':X_test,'y_test':y_test},settings.MODELS_OUTPUT_DIR + '/test_data.pkl')  
    
    logger.info('--------------------------------- Modelling - Basic Model Building -----------------------------------')
    classifiers = {} # This is an empty dictionary to store all the different classifiers
 
    # Decision Tree
    # Dictionary of the sci-kit learn decision tree parameters
    sklearn_params={
                    'criterion':'gini'
                    ,'splitter':'best'
                    ,'max_depth':10
                    ,'min_samples_split':50
                    ,'min_samples_leaf':50
                    ,'max_features':5
                    ,'compute_importances':None
                    ,'max_leaf_nodes':None
                    }
    if settings.models['Decision Tree']==True: 
        classfiers = modfuncs.decTree(X_train,y_train,X_test,y_test, classifiers,'Decision Tree',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR,pickle=True, **sklearn_params)
    
    # Logistic Regression
    # Dictionary of the sci-kit learn logistic regression parameters
    sklearn_params={
                    'penalty':'l2'
                    ,'dual':False
                    ,'tol':0.0001
                    ,'C':1.0
                    ,'fit_intercept':True
                    ,'intercept_scaling':1
                    ,'class_weight':'auto'
                    #,'random_state':None
                    }
    if settings.models['Logistic Regression']==True:
        classfiers = modfuncs.logReg(X_train,y_train,X_test,y_test, classifiers,'Logistic Regression',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR,pickle=False, **sklearn_params)
    
    # Random Forest
    # Dictionary of the sci-kit learn random forest parameters
    sklearn_params={
                    'n_estimators':30
                    ,'criterion':'gini'
                    ,'max_depth':10
                    ,'min_samples_split':50
                    ,'min_samples_leaf':100
                    ,'max_features':None
                    }
    if settings.models['Random Forest']==True:
        classfiers = modfuncs.randomForest(X_train,y_train,X_test,y_test, classifiers,'Random Forest',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR,pickle=False, **sklearn_params)  
           
    logger.info('--------------------------------- Modelling - Model Parameter Grid Searching -----------------------------------')
    # Grid Search regression model parameter evaluation
    # In each case the square brackets can take a comma separated list of the 
    # possible input values.
    param_grid = {
                    'penalty':('l1','l2')
                    ,'C':[0.01, 0.1, 0.5, 1.0]
                    ,'fit_intercept':('True','False')
                 }
    if settings.models['Logistic Regression Grid Search']==True:
        classifiers = modfuncs.logRegGridSearch(param_grid, X_train,y_train,X_test,y_test, classifiers,'Logistic Regression Grid Search',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR)
        
    # Grid Search decision tree parameter evaluation
    # In each case the square brackets can take a comma separated list of the 
    # possible input values.
    param_grid = [{
                    'criterion':['entropy','gini']
                    ,'splitter':['best']
                    ,'max_features':[5] 
                    ,'min_samples_leaf':[5,50]
                    ,'min_samples_split':[5,50]
                    ,'max_depth':[10,20]
                    ,'max_leaf_nodes':[None]
                  }]
    if settings.models['Decision Tree Grid Search']==True:
        classifiers = modfuncs.decTreeGridSearch(param_grid, X_train,y_train,X_test,y_test, classifiers,'Decision Tree Grid Search',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR)
       
    # Grid Search Random Forest
    # In each case the square brackets can take a comma separated list of the 
    # possible input values.
    param_grid = [{
                    'criterion':['gini']
                    ,'max_features':[5]
                    ,'n_estimators':[30,50]
                    ,'max_depth':[5,10,20]
                    ,'min_samples_leaf':[100,150]
                    ,'min_samples_split':[50,75]
                }]
    if settings.models['Random Forest Grid Search']==True:
        classifiers = modfuncs.ranForestGridSearch(param_grid, X_train,y_train,X_test,y_test, classifiers,'Random Forest Grid Search',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR) 
    
    #------- Non-Scoring Models -----------------------------------------
    nonProbClassifiers={} # empty dictionary to store the results
    # Stochastic Gradient Descent
    if settings.models['SGD']==True:
        nonProbClassifiers = modfuncs.SGD(X_train,y_train,X_test,y_test, nonProbClassifiers,'SGD',settings.MODELS_OUTPUT_DIR,pickle=False) 
    
    #------- Recursive Model Building -----------------------------------------
    logger.info('--------------------------------- Modelling - Recursive Feature Extraction -----------------------------------')
    # Recursively build model to determine features with stratified K fold sampling
    #logger.info('Started rfeCV logistic regression model build %s', datetime.datetime.now().time().isoformat())
    #logreg2 = sklm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
    #   fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    #skf = cv.StratifiedKFold(y, n_folds=3)
    #modelRfeLogReg2 = fs.RFECV(logreg2, step=1, cv=skf, scoring='accuracy', verbose=10)
    #modelRfeLogReg2.fit(np_data_fs, y)
    #logger.info('Model 2 parameters: %s', modelRfeLogReg2)
    #logger.info('Finished rfeCV logistic regression model build %s', datetime.datetime.now().time().isoformat())
    #get the model metrics for the test data
    #metricsRfeLogReg2 = modfuncs.modelMetrics(modelRfeLogReg2,X_test,y_test,X_train,y_train,'2. Logistic Regression2')
    # # Pickle the model
    # joblib.dump({'model':modelRfeLogReg2,'X_test':X_test,'y_test':y_test},MODELS_OUTPUT_DIR + '/modelRfeLogReg2.pkl')
    
    # Changed C=1 to C=10
    # Recursively build model to determine features
    #logger.info('Started rfeCV log reg with C=10  %s', datetime.datetime.now().time().isoformat())
    #logreg4 = sklm.LogisticRegression(penalty='l1', dual=False, tol=0.01, C=10.0, fit_intercept=True, 
    #    intercept_scaling=1, class_weight=None, random_state=None)
    #modelRfeLogReg4 = modfuncs.rfeCv(logreg2, X_train, y_train)
    #logger.info('Model 2 parameters: %s', modelRfeLogReg4)
    #logger.info('Finished rfeCV logistic regression model build %s', datetime.datetime.now().time().isoformat())
    #get the model metrics for the test data
    #metricsRfeLogReg4 = modfuncs.modelMetrics(modelRfeLogReg4,X_test,y_test,X_train,y_train,'4. Logistic Regression')
    # # Pickle the model
    # joblib.dump({'model':modelRfeLogReg4,'X_test':X_test,'y_test':y_test},MODELS_OUTPUT_DIR + '/modelRfeLogReg4.pkl')
    
    # Decision Tree RFECV
    #logger.info('Started rfeCV decision tree model build %s', datetime.datetime.now().time().isoformat())
    #modelDecTree7 = sktree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=5, min_samples_leaf=1, max_features=None, compute_importances=None, max_leaf_nodes=None)
    #modelDecTree7 = modfuncs.rfeCv(modelDecTree7, X_train, y_train)
    #logger.info('Finished rfeCV decision tree model build %s', datetime.datetime.now().time().isoformat())
    # # Pickle the model
    # joblib.dump({'model':modelDecTree7,'X_test':X_test,'y_test':y_test},MODELS_OUTPUT_DIR + '/modelDecTree7.pkl')
    
    #------- Plot ROC Curves for selected models ------------------------------
    # plot the ROC curves for the models built
    logger.info('--------------------------------- Plot ROC Curves -----------------------------------')
    modfuncs.plotRocCurve(classifiers, settings.RESULTS_OUTPUT_DIR,nonProbClassifiers)
    
if __name__ == "__main__":
	print ('Please run this script from the machine_learning_master script')
