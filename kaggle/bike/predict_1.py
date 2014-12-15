######################
# For use in Kaggles bike competiion 
#
# by David Curry, 8/10/2014
#
#####################

print '-----> Importing Modules'

import csv as csv
import numpy as np
import pandas as pd
import pylab as py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import cross_validation, metrics, linear_model, svm
from ml_modules import *


print '-----> Importing Data'

train       = pd.read_csv('train.csv')
final_test  = pd.read_csv('test.csv') 
 
# Convert datetime column into hour, year, etc.
datetime_split(train)
datetime_split(final_test)

# Store datetime before removal
date_ids    = final_test['datetime']
final_test  = final_test.drop(['datetime'], axis=1)
train       = train.drop(['datetime'], axis=1)

# remove unused outcomes
casual     = train['casual']
registered = train['registered']
train = train.drop(['casual', 'registered'], axis=1)

# move outcomes to the front
train = set_column_sequence(train, ['count'])

# convert to numpy array
train_data       = train.values
final_test_data  = final_test.values

# Cross Validiation
#cross_Validation(train_data)

 # Now split the data into x(features) and y(known outcomes)
train_data_x = train_data[:, 1:]
train_data_y = train_data[:, 0]



# ================================================================================
# Datasets are ready for SciKit Machine Learning!

print '-----> Training...'

forest = RandomForestRegressor(n_estimators=100, 
                               max_features='auto',
                               min_samples_split = 1,
                               max_depth = None
                               )



forest = forest.fit( train_data_x, train_data_y )


# ===============================================================================
# =================  Data PreSelections, Feature Selection, Covariance... =======

# pipe forest into feature slection modules
#feature_selection_trees(forest)

# Data Set covariance module
#covariance(train_data_x)

# ==================================================================================

print '-----> Predicting...'

output = forest.predict(final_test_data).astype(int)


print '\n======== Results =========\n'

#print metrics.classification_report(train_data_y, forest.predict(train_data_x) )

#print sklearn.metrics.r2_score(train_data_y, forest.predict(train_data_x) )

# ===================================================================================
# Save into Kaggle submission format

predictions_file = open("bike_results_new.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["datetime","count"])
open_file_object.writerows(zip(date_ids, output))
predictions_file.close()
print '-----> Done.'  


