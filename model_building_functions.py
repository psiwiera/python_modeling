# import libraries
from sklearn import linear_model as sklm
from sklearn import tree as sktree
from sklearn import cross_validation as cv
from sklearn import decomposition as dec
from sklearn import feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.externals.six import StringIO 
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing as prepro
from sklearn.externals import joblib
from sklearn import ensemble as en
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#import psycopg2
import pandas
from io import StringIO
#from graphviz import Graph 
#import pydot 
from inspect import getmembers
import logging
import datetime
import os
import pprint as pp # to make log entries nicer to read
import pandas as pd


#----------Training and Test Set Preparation------------------------------

def basicTrainTest(np_data, y):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Buiild.basicTrainTest')
    # Create basic training test split without stratification.
    # Does not do stratified sampling so probably shouldn't use for imbalanced class problems.
    # set the test size to be the proportion of the dataset you want to hold back from training.
    # Cross validation functionality is available in the data modeling script set.
    X_train, X_test, y_train, y_test = cv.train_test_split(np_data, y, test_size=0.30, random_state=33)

    return X_train, X_test, y_train, y_test

#def stratifiedTrainTest(np_data, y):
    # Add stratified sampling script, or do we just use Cross Validation?

    #   return np_data

def kfoldTrainingSet(np_data_fs,y,n_folds=10):
    # Use stratified k fold sampling to create training and test set.
    skf = cv.StratifiedKFold(y, n_folds=n_folds)
    for train_index, test_index in skf:
        X_train, X_test = np_data_fs[train_index], np_data_fs[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    return X_train,y_train,X_test,y_test
    
def featureSelection(X, y, var_results,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Buiild.featureSelection')
    # Downselect variables based upon feature selection.
    # When training using a large number of variables can use this for an initial downselection
    # which will help improve model building performance. percentile selects the vars within the top x
    # percentile based on an evaluation metric.
    print('numpy array shape')
    print(X.shape)
    
    chi2vals = fs.chi2(X, y)
    
    print('numpy array test shape')
    print(X.shape)
    #chi2Dict, chi2_pvalDict = addVarNamestoList(var_names, chi2vals)
    #logger.info('Zipped Chi2 values inc PP: %s', pp.pformat(chi2Dict))
    #logger.info('Variable Chi2 p-values: %s', pp.pformat(chi2_pvalDict))
    
    featsel = fs.SelectPercentile(fs.chi2, percentile=50)
    X_fs = featsel.fit_transform(X, y)
    
    #add which variables have been selected and which ones removed by feature selection to var_results
    # add chi2 and p values to data frame.
    feat_results = featsel.get_support()
    var_results['feat_selection'] = feat_results
    var_results['featsel_chi2'] = chi2vals[0]
    var_results['featsel_chi2_pvals'] = chi2vals[1]
    print(var_results)
    # save the results to a csv file and update log.
    var_results.to_csv(RESULTS_OUTPUT_DIR + 'feature_selection.csv', sep=',')
    logger.info('Feature selection results saved to csv')
    
    c = var_results.loc[var_results['feat_selection']==False]
    print(c.shape)
    # Pickle the results of the feature selection
    joblib.dump(var_results,MODELS_OUTPUT_DIR + '/feature_selection.pkl')  
    
    return X_fs, var_results
    
def addVarNamestoList(var_names, np_array):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.addVarNmestoList')
    # function combines data from list
    chi2Dict = dict(zip(var_names, np_array[0]))
    chi2_pvalDict = dict(zip(var_names, np_array[1]))
    
    return chi2Dict, chi2_pvalDict
#----------Var Transformation----------------------------------------------
def scaler(np_data):
    # normalise and scale variables (helps algorithms to perform better)
    minMaxScaler = prepro.MinMaxScaler()
    np_data_scaled = minMaxScaler.fit_transform(np_data)
    
    return np_data_scaled
    
    
#----------Model functions-------------------------------------------------
def logReg(X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,pickle=False, **kwargs):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.logReg')
    
    # build the classifier
    model,classifiers = calcClfModel(sklm.LogisticRegression(**kwargs),X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)  
    
    
    logger.info('Started Log Reg 5 model build %s', datetime.datetime.now().time().isoformat())
    # logger.info('Optimum Log Reg Parameters: %s', pp.pformat(modelLogReg5.coef_))
    # #get the model metrics for the test data
    
   
    

    logger.info('Logistic Regression Model Results --------------------------')
    logger.info('Model coefficients: %s', pp.pformat(model.coef_))
    logger.info('Model intercept: %s', pp.pformat(model.intercept_))

    return classifiers

    
def randomForest(X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,pickle=False, **kwargs):
    # set logging
    logger = logging.getLogger('Master.Models_Build.Functions.randomForest')
    
    # build the classifier
    model,classifiers = calcClfModel(en.RandomForestClassifier(**kwargs),X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)  
    
    return classifiers
    
def decTree(X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,pickle=False, **kwargs):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.decTree')

    # build the classifier
    model,classifiers = calcClfModel(sktree.DecisionTreeClassifier(**kwargs),X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)    
    
    # plot the decision tree
    plotDecTree(model,RESULTS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.dot')
    logger.info('Decision tree details: %s', classifiers[label])
    
    logger.info('Decision Tree Model Results --------------------------------')
    logger.info('Feature importances: %s', pp.pformat(model.feature_importances_))

    return classifiers

def calcClfModel(clf,X_train,y_train,X_test,y_test, label,classifiers,MODELS_OUTPUT_DIR, pickle=False):
    # train the model
    model = clf.fit(X_train,y_train)
    # calculate the model metrics
    metrics = modelMetrics(model,X_test,y_test,X_train,y_train,label)
    
    classifiers[label] = metrics
    
    if pickle:
        # Pickle the model and the test data (to calc metrics later on)
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl')
    
    return model,classifiers
    
#---------Model Functions Recursive Feature Selection with Cross Validation-
def rfeCv(modEstimator,X,y):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.rfeCv')
    # uses the rfecv functionality to identify the optimum number of variables (features)
    # along with cross validation to determine the most optimum model.
    rfecv = fs.RFECV(estimator=modEstimator, step=1, cv=3, 
        scoring='accuracy', verbose=10)
    rfecv.fit(X, y)

    logger.info("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    return rfecv

# Use cross validation to build model.
def crossValScore(unbuilt_model,X,y):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.crossValScore')
    cross_val_score(unbuilt_model, X, y, cv=10) 

#---------- Model Functions with Grid Search Parameter Optimisations------
def logRegGridSearch(param_grid, X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,score='roc_auc',pickle=False):
    # set logging
    logger = logging.getLogger('Master.Models_Build.Functions.logRegGridSearch')
    
    # set the estimator
    modEstimator= sklm.LogisticRegression(class_weight='auto')
    
    # run the grid search
    # Note - setting n_jobs > 1 will cause a logging error.
    gd_clf = GridSearchCV(modEstimator, param_grid, n_jobs=1, refit=True, verbose=10,scoring=score)
    
    # Print the results to the log file
    # logger.info('Optimum Random Forest Parameters: %s', pp.pformat(modelRanFor13.best_params_))
    # logger.info('Finished Random Forest Grid Search model build %s', datetime.datetime.now().time().isoformat())
    
    # get the metrics for the output
    classifiers = runGridSearchFit(gd_clf,X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR,pickle)
    
    logger.info('Optimum Logistic Regression Parameters: %s', pp.pformat(gd_clf.best_estimator_))
    
    return classifiers
    
def decTreeGridSearch(param_grid, X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,score='roc_auc',pickle=False):
    # set logging
    logger = logging.getLogger('Master.Models_Build.Functions.decTreeGridSearch')
    
    # set the estimator
    modEstimator = sktree.DecisionTreeClassifier()
    
    # run the grid search
    # Note - setting n_jobs > 1 will cause a logging error.
    gd_clf = GridSearchCV(modEstimator, param_grid, n_jobs=1, refit=True, verbose=10,scoring=score)
    
    # Print the results to the log file
    # logger.info('Optimum Random Forest Parameters: %s', pp.pformat(modelRanFor13.best_params_))
    # logger.info('Finished Random Forest Grid Search model build %s', datetime.datetime.now().time().isoformat())
    
    # get the metrics for the output
    classifiers = runGridSearchFit(gd_clf,X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR,pickle)
    
    logger.info('Optimum Decision Tree Parameters: %s', pp.pformat(gd_clf.best_estimator_))
    
    return classifiers
    

def ranForestGridSearch(param_grid, X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,score='roc_auc',pickle=False):
    # set logging
    logger = logging.getLogger('Master.Models_Build.Functions.ranForestGridSearch')
    
    # set the estimator
    modEstimator = en.RandomForestClassifier()
    
    # run the grid search
    # Note - setting n_jobs > 1 will cause a logging error.
    gd_clf = GridSearchCV(modEstimator, param_grid, n_jobs=1, refit=True, verbose=10, scoring=score)
    
    # Print the results to the log file
    # logger.info('Optimum Random Forest Parameters: %s', pp.pformat(modelRanFor13.best_params_))
    # logger.info('Finished Random Forest Grid Search model build %s', datetime.datetime.now().time().isoformat())
    
    
    # get the metrics for the output
    classifiers = runGridSearchFit(gd_clf,X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR,pickle)
    
    logger.info('Optimum Random Forest Parameters: %s', pp.pformat(gd_clf.best_params_))
    
    return classifiers

def runGridSearchFit(gd_clf,X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR,pickle):
    # set logging
    logger = logging.getLogger('Master.Models_Build.Functions.runGridSearchFit')
    
    # fit the model
    model = gd_clf.fit(X_train,y_train)
    
    # get the metrics
    metrics = modelMetrics(model,X_test,y_test,X_train,y_train,label)
    classifiers[label] = metrics
    
    if pickle:
        # Pickle the model and the test data (to calc metrics later on)
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl')
        
    return classifiers
    
    

#-------------Classifiers with no predict probabilities function---------
def nonProbMetrics(model,X,y):
    predicted = model.predict(X)
    # predicted is an array of predictions, compare this to y to get the false positive rate
    # and the true positive rate to put a point on the ROC curve
    # Note this requires the positive to be 1 and negative to be 0
    fp = sum((predicted != y) * predicted)
    tp = sum((predicted == y) * predicted)
    predicted_invert=predicted
    predicted_invert[predicted_invert==1]=2
    predicted_invert[predicted_invert==0]=1
    predicted_invert[predicted_invert==2]=0
    
    fn = sum((predicted != y) * predicted_invert)
    tn = sum((predicted == y) * predicted_invert)
    
    fpr = fp/float(fp+tn)
    tpr = tp/float(tp+fn)
    
    return fpr,tpr
    
    


def SGD(X_train,y_train,X_test,y_test, nonProbClassifiers,label,MODELS_OUTPUT_DIR,pickle=False, **kwargs):
    logger = logging.getLogger('Master.Models_Build.Functions.SGD')
    logger.info('Started Grad Desc model build %s', datetime.datetime.now().time().isoformat())
    
    model = sklm.SGDClassifier(**kwargs)
    model = model.fit(X_train,y_train)
    #get the model metrics for the test data
    #metrics = modfuncs.modelMetrics(model, X_test, y_test,X_train,y_train, label)
    fpr_test,tpr_test= nonProbMetrics(model,X_test,y_test)
    fpr_train,tpr_train =nonProbMetrics(model,X_train,y_train)
    
    nonProbClassifiers[label] = {'tpr_test':tpr_test,'fpr_test':fpr_test,'tpr_train':tpr_train,'fpr_train':fpr_train}
    
    # Pickle the model
    if pickle:
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl')  
    
    return nonProbClassifiers
    
    
#---------- Visualisation Functions---------------------------------------
def plotDecTree(tree, filename):
    # #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.plotDecTree')
    # # saves the model in dot format to visualise using GraphViz application
    sktree.export_graphviz(tree, out_file = filename) 
    logger.info('Decision tree produced and saved as dot file')


def calcModelMetrics(model,X,y):
    # apply the model to the input test data
    test_probas_ = model.predict_proba(X)

    # Compute ROC curve and area under the curve
    test_fpr, test_tpr, test_thresholds = roc_curve(y,test_probas_[:,1])
    test_roc_auc = auc(test_fpr,test_tpr)
    return test_fpr,test_tpr,test_roc_auc
    
    
# Define a function to calculate the testing metrics associated with 
# the test (as defined by the input) data.
# the input model is an sklearn model.fit() object that has been trainied
# X_test, y_test are the test vector and class to use
# the label is a label to use to refer to the model
# and the function returns a dictionary  object with the details of the results
def modelMetrics(model, X_test, y_test, X_train,y_train, label):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.modelMetrics')
    
    # Apply the model to the test data
    test_fpr,test_tpr,test_roc_auc = calcModelMetrics(model,X_test,y_test)
    
    # And to the training data
    train_fpr,train_tpr,train_roc_auc = calcModelMetrics(model,X_train,y_train)
    
    print("Area under ROC curve for model "+label+" test, train: %f" % test_roc_auc,train_roc_auc)
    
    # build a dictionary with the relevant output from the model
    outputDict = {
                'label':label,
                'test_false_positive':test_fpr,
                'test_true_positive':test_tpr,
                'test_ROC_auc':test_roc_auc,
                'train_false_positive':train_fpr,
                'train_true_positive':train_tpr,
                'train_ROC_auc':train_roc_auc
                }
    return outputDict

# Define function to Plot ROC curve
def plotRocCurve(classifiers, RESULTS_OUTPUT_DIR,nonProbClassifiers={}):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('resulst.Models_Build.plotRocCurve')

    pl.clf()
    #pl.plot(log_fpr, log_tpr, label='Logistic Regression ROC curve (area = %0.2f)' % log_roc_auc)

    pl.figure(1)
    pl.subplot(121)
    
    for key,value in classifiers.items():
        pl.plot(value['test_false_positive'], value['test_true_positive'], label=str(value['label']) + ' (area = %0.2f)' % value['test_ROC_auc'])
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Testing set ROC Curve')
    pl.legend(loc="lower right",fontsize='x-small')
    
    # if we have additional fpr/tpr points to plot lets plot them
    if len(nonProbClassifiers)>0:
        for key,cl in nonProbClassifiers.items():
            pl.scatter(cl['fpr_test'],cl['tpr_test'],marker='o')
            pl.annotate(key,xy=(cl['fpr_test'],cl['tpr_test']),xytext=(-20,20),textcoords='offset points',ha='right',va='bottom',bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0'))
    
    pl.subplot(122)
    for key,value in classifiers.items():
        pl.plot(value['train_false_positive'], value['train_true_positive'], label=str(value['label']) + ' (area = %0.2f)' % value['train_ROC_auc'])
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Training set ROC curve')
    pl.legend(loc="lower right",fontsize='x-small')
    
    # if we have additional fpr/tpr points to plot lets plot them
    #if len(nonProbClassifiers)>0:
        
    
    savepath = os.path.join(RESULTS_OUTPUT_DIR, 'ROC_Curve.png')
    pl.savefig(savepath)
    pl.show()
    pl.close()
    
def test_roc_curve(model, X_test, y_test):
    probas_ = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print('ROC Area: %f' %roc_auc)
    
    plt.clf()
    pl.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' %roc_auc)
    pl.plot([0,1], [0,1], 'k--')
    pl.xlim([0.0,1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('FPR')
    pl.ylabel('TPR')
    pl.title('ROC Curve')
    pl.show()
