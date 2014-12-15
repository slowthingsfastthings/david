######################
# For use in Kaggles bike competiion
#
# by David Curry, 8/10/2014
#
#####################

print '-----> Importing Modules'

import numpy as np
import pandas as pd
import pylab as py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ml_modules import *


print '-----> Importing Data'

#train       = pd.read_csv('train.csv')
#final_test  = pd.read_csv('test.csv')

print train.info()
print final_test.info()

# get test ids
ids         = final_test['Id'] 
final_test  = final_test.drop(['Id'], axis=1)
train       = train.drop(['Id'], axis=1)

# check for empty entries
for col in final_test:
    if len( final_test[final_test[col].isnull()] ) > 0: print col

for col in train:
    if len( train[train[col].isnull()] ) > 0: print col
    


# ====== Make some new variables =========
#train = square_features(train, 'Cover_Type')

#train = cross_features(train, 'Cover_Type')

#print train.info()

# ========================================


# move known outcomes to the front    
train = set_column_sequence(train, ['Cover_Type'])


# convert to numpy array
final_test_data = final_test.values
train_data1     = train.values

# Now split the data into x(features) and y(known outcomes)
train_data_x = train_data1[:, 1:]
train_data_y = train_data1[:, 0]

# ====== Preproccessing ===================
#scale the data to have zero mean and unit variance.  Only necessary with SVM's
#train_data_x = preprocessing.scale(train_data_x)
#test_data_x  = preprocessing.scale(test_data_x)
 

# ================================================================================
# Datasets are ready for SciKit Machine Learning!

print '-----> Training...'


# ============= Cross-Validation ==============
# Feed train_data1(unsplit data) into module.  Outcomes are first column
cross_Validation(train_data1)


# Move into final fit and outcomes
forest = RandomForestClassifier(n_estimators=60,
                                max_features = 4,
                                #min_samples_split=1,
                                #compute_importances=all
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

#print sklearn.metrics.accuracy_score(test_data_y, forest.predict(test_data_x) )


# ===================================================================================
# Save into Kaggle submission format

predictions_file = open("forrest_results_random_forest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Cover_Type"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()



print '-----> Done.' 

