########################################################################
##  ml_modules.py
##  A collection of useful modules to be used as plugins in larger code
##
##  by David Curry, 8/10/2014
##
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import sklearn.covariance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn import cross_validation, preprocessing


def cross_Validation(dataset):
    
    '''
    Takes in numpy array and uses scikit to perform several cross validation tests. PLots and results are printed
    See: scikit.com
    '''

    print '\n\n============ Cross-Validation =============\n'

    # split training set into a test and train part.  80% train, 20% Test
    train_data, test_data = cross_validation.train_test_split(dataset, test_size=0.05, random_state=42)
    
    # Now split the data into x(features) and y(known outcomes)
    train_data_x = train_data[:, 1:]
    train_data_y = train_data[:, 0]

    test_data_x = test_data[:, 1:]
    test_data_y = test_data[:, 0]
    
    
    #scale the data to have zero mean and unit variance.  Only necessary with SVM's
    #std_scale = preprocessing.StandardScaler().fit(train_data_x)
    std_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_data_x)
    train_data_x = std_scale.transform(train_data_x)
    test_data_x  = std_scale.transform(test_data_x)
    
    # Vary model complexity and plot train/test error against each other
    estimator_list         = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] 
    max_feature_list       = [1, 2, 3, 4, 5, 6, 'auto', 'sqrt', None]
    min_samples_split_list = [1, 2, 3, 4, 5, 6]
    
    for x in estimator_list:
    #for x in max_feature_list:
    #for x in min_samples_split_list:

        print '---> Training with', x, 'estimators'
        
        '''
        forest = RandomForestClassifier(n_estimators = x,
                                        #max_features = None,
                                        #min_samples_split=1
                                        )

        # Fit the data
        forest = forest.fit( train_data_x, train_data_y )
        '''
        
        # ada boost
        forest = AdaBoostClassifier(n_estimators = x,
                                    )
        

        forest = forest.fit( train_data_x, train_data_y )
        
        # Compute Errors    
        test_accuracy  = sklearn.metrics.accuracy_score(test_data_y, forest.predict(test_data_x) )
        train_accuracy = sklearn.metrics.accuracy_score(train_data_y, forest.predict(train_data_x) ) 
        
        
        

        
        if x is 'auto': x = 7
        if x is 'sqrt': x = 8
        # fill plots
        plt.plot(x, test_accuracy, 'bo')
        plt.plot(x, train_accuracy, 'bo')
        
        
    plt.show()
    
    
    print '\n=========== End Cross-Validation ===========\n'
    

def feature_selection_trees(forest):
    
    ''' 
    Takes in a scikit forest object and performs several feature selection tests.
    See: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
    '''
    
    print '\n\n============ Feature Selection =============\n'
    
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print 'Feature ranking:'
    
    for f in range(len(indices)):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices],
            color="r", align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()
        
    

        
    print '\n============ End Feature Selection =============\n'




def covariance(data_x):

     print '\n=============== Covariance =======================\n'
     
     emp_cov = (np.dot(data_x.T, data_x)).astype(np.float32)
     
     vmax = emp_cov.max()
     plt.imshow(emp_cov, interpolation='nearest', #vmin=-vmax, vmax=vmax, 
                cmap=plt.cm.RdBu_r)
     plt.xticks(())
     plt.yticks(())
     plt.title('Empirical Covariance')
      
     #plt.show()

     new_cov = sklearn.covariance.empirical_covariance(data_x)
     
     plt.imshow(new_cov, interpolation='nearest', #vmin=-vmax, vmax=vmax,
                cmap=plt.cm.RdBu_r)
     plt.xticks(())
     plt.yticks(())
     plt.title('Empirical Covariance')

     plt.show()


     print '\n============ End Covariance =============\n'



def set_column_sequence(dataframe, seq):
    '''Takes a dataframe and a subsequence of its columns, returns dataframe with seq as first columns'''

    cols = seq[:] # copy so we don't mutate seq

    for x in dataframe.columns:

        if x not in cols:
            
            cols.append(x)

    return dataframe[cols]




def square_features(dataset, outcomes):

    '''
    Takes in pandas dataset and name of outcome column
    Squares all features in a dataseta and returns with new columns appended.
    '''
    
    for col in dataset:
        
        if col == outcomes: continue
        
        temp_name = col+'_sqr'

        dataset[temp_name] = dataset[col] * dataset[col]

    return dataset


def cross_features(dataset, outcomes):

    '''
    Takes in pandas dataset and name of outcome column
    Creates cross multiplication terms all features in a dataseta and returns with new columns appended.
    '''

    temp_dataset = dataset

    for x,y in enumerate(temp_dataset):
        for i,j in enumerate(temp_dataset):

            if x <= i: continue
            
            if y == outcomes or j == outcomes: continue
            
            temp_name = y + '_' + j
        
            dataset[temp_name] = dataset[y] - dataset[j]

    return dataset




