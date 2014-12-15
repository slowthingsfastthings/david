# For use in Kaggles bike competiion
#
# by David Curry, 8/10/2014
#
#  Modules to Run: Specify when calling function(ie. predict.py train)
#       - plot_features
#       - crossVal
#       - optimize
#       - train
#       - predict
#
#
##################################################


print '-----> Importing Modules'

import csv as csv
import numpy as np
import pandas as pd
import pylab as py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn import cross_validation, metrics, linear_model, svm, preprocessing
from ml_modules import *

print '-----> Importing Data'

train = pd.read_csv('/Users/HAL3000/Dropbox/coding/data/kaggle/forest/train.csv')
final_test  = pd.read_csv('/Users/HAL3000/Dropbox/coding/data/kaggle/forest/test.csv')


# ==== Choose what Modules to Run =========================
plot_features, crossVal, training, optimize, predict = False, False, False, False, False

# Set the module level
if len(sys.argv) is 1:
    sys.exit( '-----> Please specify a module to Run.  Choices are: \n       plot_features, crossVal, training, optimize, predict')
else: module = sys.argv[1]

if module == 'plot_features':
    plot_features = True

if module == 'crossVal':
    crossVal = True

if module == 'optimize':
    optimize = True

if module == 'training':
    training = True

if module == 'predict':
    predict = True

# =======================================================


# get test ids
ids         = final_test['Id'] 
final_test  = final_test.drop(['Id'], axis=1)
train       = train.drop(['Id'], axis=1)


# merge soil type columns. Add to existing dataframe and erase old columns
newCol = pd.Series('', index=train.index)
for count, col in enumerate(train.loc[:, 'Soil_Type1':'Soil_Type40']):
    newCol[ train[col] == 1] = count + 1
    train = train.drop([col], axis=1)

train['Soil_Type_All'] = newCol.astype(int)

newCol2 = pd.Series('', index=train.index)
for count, col in enumerate(train.loc[:, 'Wilderness_Area1':'Wilderness_Area4']):
    newCol2[ train[col] == 1] = count + 1
    train = train.drop([col], axis=1)

train['Wild_Area_All'] = newCol2.astype(int)


# Now the final test set
newCol = pd.Series('', index=final_test.index)
for count, col in enumerate(final_test.loc[:, 'Soil_Type1':'Soil_Type40']):
    newCol[ final_test[col] == 1] = count + 1
    final_test = final_test.drop([col], axis=1)

final_test['Soil_Type_All'] = newCol.astype(int)

newCol2 = pd.Series('', index=final_test.index)
for count, col in enumerate(final_test.loc[:, 'Wilderness_Area1':'Wilderness_Area4']):
    newCol2[ final_test[col] == 1] = count + 1
    final_test = final_test.drop([col], axis=1)

final_test['Wild_Area_All'] = newCol2.astype(int)




# move known outcomes to the front    
train = set_column_sequence(train, ['Cover_Type'])

# Quick look at data    
#print train.head()
#print train['Cover_Type'].value_counts()
#print final_test.info()




# ============= Plot Features w/Pandas =========
if plot_features:    
    
    print '-----> Performing Feature Tests...'
    
    plot_feature(train)
    
    
    
# ============= Cross-Validation ==============
if crossVal:
    
    print '-----> Performing Cross Validation...'
    
    # test/train accuracy for trees
    #accuracy_trees(train)
    
    # accuracy for SVMs
    accuracy_svm(train)
    
    # Make ROC plots
    #roc(train, GradientBoostingClassifier)
    
    # Make confusion matrix for multiclass classification
    #confusion_Matrix(train)
    

# ============= Parameter Optimization ============
if optimize: 
    
    print '-----> Performing Parameter Optimization...'
    
    train_optimizer(train)
    
    
    
    

# ============= Datasets are ready for SciKit Machine Learning! ================
if predict:
    
    print '-----> Final Training...'
    
    # convert to numpy array
    final_test_data = final_test.values
    train_data1     = train.values
    
    # Now split the data into x(features) and y(known outcomes)
    train_data_x = train_data1[:, 1:]
    train_data_y = train_data1[:, 0]
    
    # Move into final fit and outcomes
    forest = GradientBoostingClassifier(n_estimators = 100,
                                        max_features = 'auto',
                                        min_samples_split = 1,
                                        max_depth = 5,
                                        learning_rate = 0.5
                                        )
    
    forest = forest.fit( train_data_x, train_data_y )

    print '-----> Predicting...'

    output = forest.predict(final_test_data).astype(int)

    # Save into Kaggle submission format
    predictions_file = open("forrest_results_new.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","Cover_Type"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()


print '-----> Done.' 

