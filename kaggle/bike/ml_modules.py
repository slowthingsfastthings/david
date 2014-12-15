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
from datetime import datetime
from sklearn.grid_search import GridSearchCV

def cross_Validation(dataset):
    
    '''
    Takes in numpy array and uses scikit to perform several cross validation tests. PLots and results are printed
    See: scikit.com
    '''

    print '\n\n============ Cross-Validation =============\n'

    # split training set into a test and train part.  80% train, 20% Test
    train_data, test_data = cross_validation.train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Now split the data into x(features) and y(known outcomes)
    train_data_x = train_data[:, 1:]
    train_data_y = train_data[:, 0]

    test_data_x = test_data[:, 1:]
    test_data_y = test_data[:, 0]
    
    # define the parameter search space
    search_space = [
        { 'max_features': [1, 2, 3, 4, 5, 'auto', 'sqrt', None],
          'min_samples_split': [1, 2, 3, 4, 5],
          'max_depth': [1, 2, 3, 4, 5, None] }
        ]


    # Define the type of algrothm to employ
    model = RandomForestRegressor(n_estimators = 10)

    print '=== Tuning hyper-parameters ==='
    
    '''
    cv = GridSearchCV(
        model, 
        search_space, 
        scoring='r2')
    
    cv.fit(train_data_x, train_data_y)
    
    print 'Best parameters found on training set:'
    print cv.best_estimator_
    '''
    '''
    print '\n\nGrid scores on training set:'
    for params, mean_score, scores in cv.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    '''

     # if performing classification print a report out
    '''
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_data_y, clf.predict(test_data_x)
    print(classification_report(y_true, y_pred))
    '''
    
    # Vary model complexity and plot train/test error against each other
    estimator_list         = [1, 5, 10, 20, 30, 40, 50, 60] 
    
    for x in estimator_list:

        print '---> Training with', x, 'estimators'
        
        forest = RandomForestRegressor(n_estimators = x,
                                        max_features = 'auto',
                                        min_samples_split = 1,
                                        max_depth = None
                                        )
        
        # Fit the data
        forest = forest.fit( train_data_x, train_data_y )
        
        # Compute Errors    
        test_accuracy  = sklearn.metrics.mean_squared_error(test_data_y, forest.predict(test_data_x) )
        train_accuracy = sklearn.metrics.mean_squared_error(train_data_y, forest.predict(train_data_x) ) 
        
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
    '''Takes a dataframe and a subsequence of its columns, returns dataframe with seq as first columns.
       set_column_sequence(dataframe, ['outcomes']) 
    '''
    
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


def datetime_split(dataframe):

    '''
    Takes in dataframe of with a 'datetime' column of the form Year-month-day hour:minute:second
    and returns new columns for year, month, weekday, hour.
    '''
    
    date_format_str = '%Y-%m-%d %H:%M:%S'
    
    def parse_date(date_str):
        """Return parsed datetime tuple"""

        d = datetime.strptime(date_str, date_format_str)

        return {'year': d.year, 'month': d.month, 'day': d.day,
                'hour': d.hour, 'weekday': d.weekday()}

    def process_date_column(data_sframe):
        """Split the 'datetime' column of a given data frame"""
        
        parsed_date = data_sframe['datetime'].apply(parse_date)
        
        data_sframe['year']    = ''
        data_sframe['month']   = ''
        data_sframe['day']     = ''
        data_sframe['hour']    = ''
        data_sframe['weekday'] = ''

        for row in range(len(parsed_date)):
            for col in ['year', 'month', 'day', 'hour', 'weekday']:
                data_sframe[col][row] = parsed_date[row][col]
            

    return process_date_column(dataframe)
