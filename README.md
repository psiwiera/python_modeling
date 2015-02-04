python_modeling
===============

These scripts provide an end to end work flow for building risk scoring models using scikit-learn and pandas libraries. Other libraries are also used but those two act as the core.

The scripts are setup to be run from the Machine Learning Master script which calls three main stages, data load, data preprocessing and model building. data load has routines for using postgresql to load data into a pandas data frame. preprocessing focusses on data manipulation whilst the data is in a pandas data frame. model building focusses on final data manipulation using scikit-learn functions when the data is in a np-array, a variety of model building functions and model evaluation.

A script is also included to publish the final model to a web container for execution.

The scripts have been developed using an example data set freely available online.
