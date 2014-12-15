########################################################################
##  ml_modules.py
##  A collection of useful modules to be used as plugins in larger code
##
##  by David Curry, 8/10/2014
##
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
import sklearn.covariance
from sklearn import cross_validation, preprocessing
from datetime import datetime
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier
import pandas as pd




def plot_feature(dataframe):
    
    '''
    Create plots of input features.  Takes in pandas dataframe
    '''

    print '-----> Perfroming Feature Tests...'

    for col in dataframe:

        # basic feature plots
        train[col].hist()
        plt.savefig('features/basic/'+col+'.png')
        plt.clf()

        # Target Temp plots
        #train.plot(kind='hexbin', x=col, y='Cover_Type', gridsize=25)

        hist, xedges, yedges = np.histogram2d(train[col], train['Cover_Type'], bins=40)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
        plt.imshow(hist.T, extent=extent, interpolation='nearest', origin='lower', aspect='auto')

        plt.savefig('features/temp/'+col+'_temp.pdf')
        plt.clf()

    # end column loop

    # density plots
    print '-----> Density Plots'
    train.ix[np.random.choice(train.index, 1000)]
    andrews_curves(train, class_column='Cover_Type')
    plt.savefig('features/andrews_curves.pdf')

    # covariance
    train_data = train.values
    #covariance(train_data)

    # Mutual Information
    mutual_info(train, train['Cover_Type'])

    raw_input('Press Return to Continue...')


# end plot_features ==================



def accuracy_svm(dataframe):
    
    
    '''
    Plots accuracy of SVM model as a function of model complexity
    Takes in pandas dataframe and uses sciKit.
    '''

    print '-----> Peforming SVM Accuracy Tests...'

    # convert pandas to numpy
    dataset = dataframe.values
    
    # split training set into a test and train part.  80% train, 20% Test
    train_data, test_data = cross_validation.train_test_split(dataset, test_size=0.4, random_state=42)

    # Now split the data into x(features) and y(known outcomes)
    train_data_x = train_data[:, 1:]
    train_data_y = train_data[:, 0]

    test_data_x = test_data[:, 1:]
    test_data_y = test_data[:, 0]

    # scale the data for SVM input
    test_data_x, train_data_x = preprocessing.scale(test_data_x), preprocessing.scale(train_data_x)


    #  ==== choose 1 v 1, or 1 v rest  =========
    #algorithm = SVC; title = 'SVM:one_v_one'

    algorithm = LinearSVC; title = 'SVM:one_v_rest'
    # ==========================================

    
    if algorithm is SVC:
        one_v_one = algorithm( C=1.0, 
                               cache_size=1000
                               )
    
        one_v_one.fit(train_data_x, train_data_y)
        
        one_v_one.predict(test_data_x)
    
        print 'SVM(one vs one) Accuracy Score:', sklearn.metrics.accuracy_score(test_data_y, one_v_one.predict(test_data_x) )


    if algorithm is LinearSVC:
        one_v_rest = algorithm( C=1.0 )

        one_v_rest.fit(train_data_x, train_data_y)

        one_v_rest.predict(test_data_x)

        print 'SVM(one vs rest) Accuracy Score:', sklearn.metrics.accuracy_score(test_data_y, one_v_rest.predict(test_data_x) )


# ==== End error_SVM



def roc(dataframe, algorithm):
    
    '''
    Creates ROC curve plots for classification.  Can also perform multiclass ROC curves.  
    Takes in pandas dataframe and uses sciKit.
    '''
    
    print '-----> Creating ROC Plots...'
    
    # convert pandas to numpy
    dataset = dataframe.values
    
    # Manually enter the target lables
    target_names = [1,2,3,4,5,6,7]
    
    # split training set into a test and train part.  80% train, 20% Test
    train_data, test_data = cross_validation.train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Now split the data into x(features) and y(known outcomes)
    train_data_x = train_data[:, 1:]
    train_data_y = train_data[:, 0]
    
    test_data_x = test_data[:, 1:]
    test_data_y = test_data[:, 0]
    
    # Binarize the output
    train_data_y = label_binarize(train_data_y, classes = target_names)
    test_data_y  = label_binarize(test_data_y, classes = target_names)
    n_classes = test_data_y.shape[1]
    
    # Learn to predict each class against the other
    #classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    
    forest = RandomForestClassifier(n_estimators = 100, n_jobs=-1)

    # Fit the data
    forest = forest.fit( train_data_x, train_data_y )

    # Fit the data and get the classification score for each event
    classifier_score = forest.score(test_data_x, test_data_y)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_data_y[:, i], classifier_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), classifier_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()




def accuracy_trees(dataframe):
    
    '''
    Made for BDTs or random forests
    Takes in pandas dataframe and uses scikit to perform several cross validation tests. PLots and results are printed
    See: scikit.com
    '''
    
    #algorithm = GradientBoostingClassifier; title = 'GradientBoostingClassifier'
    
    #algorithm = GradientBoostingRegressor; title = 'GradientBoostingRegressor'
    
    #algorithm = RandomForestClassifier; title = 'RandomForestClassifier'
    
    #algorithm = RandomForestRegressor; title = 'RandomForestRegressor'
    
    plt.ion()
    
    # convert pandas to numpy array
    dataset = dataframe.values

    # split training set into a test and train part.  80% train, 20% Test
    train_data, test_data = cross_validation.train_test_split(dataset, test_size=0.4, random_state=42)
    
    # Now split the data into x(features) and y(known outcomes)
    train_data_x = train_data[:, 1:]
    train_data_y = train_data[:, 0]
    
    test_data_x = test_data[:, 1:]
    test_data_y = test_data[:, 0]
    
    
    estimator_list = [1, 10, 100, 1000, 5000] 
    #estimator_list = [10, 100, 1000, 5000, 10000]
    train_accuracy     = []
    test_accuracy      = []
    random_accuracy    = []
    most_freq_accuracy = []
    
    # for trees
    for x in estimator_list:
        
        print '---> Training with', x, 'estimators'
        
        #  ================ Gradiet Boosting classifier ====================
        if algorithm is GradientBoostingClassifier:
            forest = algorithm(n_estimators = x,
                               max_features = 'auto',
                               min_samples_split = 1,
                               max_depth = 5,
                               learning_rate = 0.5
                               )
            
            # Fit the data
            forest = forest.fit(train_data_x, train_data_y)
            
            test_accuracy.append(sklearn.metrics.accuracy_score(test_data_y, forest.predict(test_data_x) ))
            train_accuracy.append(sklearn.metrics.accuracy_score(train_data_y, forest.predict(train_data_x) ))
            
        # ======================================================================
            
        
        # ==================== Gradiet Boosting regressor ========================
        if algorithm is GradientBoostingRegressor:
            forest = algorithm(n_estimators = x,
                               loss='ls',
                               max_features = 'auto',
                               min_samples_split = 1,
                               max_depth = 5,
                               learning_rate = 0.5
                               )
            
            # Fit the data
            forest = forest.fit( train_data_x, train_data_y )
            
            #test_accuracy.append(sklearn.metrics.r2_score(test_data_y, forest.predict(test_data_x) ))
            #train_accuracy.append(sklearn.metrics.r2_score(train_data_y, forest.predict(train_data_x) ))

            # Also take floor of regressed value to classify the estimation
            test_accuracy.append(sklearn.metrics.accuracy_score(test_data_y, np.rint(forest.predict(test_data_x)) ))
            train_accuracy.append(sklearn.metrics.accuracy_score(train_data_y, np.rint(forest.predict(train_data_x)) ))

        # =====================================================================
            


        # ==================== random forest classifier ========================

        if algorithm is RandomForestClassifier:
            forest = algorithm(n_estimators = x,
                               max_features = None,
                               min_samples_split = 2,
                               max_depth = 10,
                               n_jobs=-1
                               )
            
            # Fit the data
            forest = forest.fit( train_data_x, train_data_y )
            
            test_accuracy.append(sklearn.metrics.accuracy_score(test_data_y, forest.predict(test_data_x) ))

            train_accuracy.append(sklearn.metrics.accuracy_score(train_data_y, forest.predict(train_data_x) ))
            
            # Also make dummy estimators for model comparison
            most_freq = DummyClassifier(strategy='most_frequent',random_state=0).fit(train_data_x, train_data_y)
            random    = DummyClassifier(strategy='uniform',random_state=0).fit(train_data_x, train_data_y)
            
            most_freq_accuracy.append(most_freq.score(test_data_x, test_data_y))
            random_accuracy.append(random.score(test_data_x, test_data_y))



        # =================== end random forest classifier ========================


        
        # random forest regressor
        if algorithm is RandomForestRegressor:
            forest = algorithm(n_estimators = x,
                               max_features = None,
                               min_samples_split = 2,
                               max_depth = 10,
                               n_jobs=-1
                               )

            # Fit the data
            forest = forest.fit( train_data_x, train_data_y )
            
            test_accuracy.append(sklearn.metrics.r2_score(test_data_y, forest.predict(test_data_x) ))
            train_accuracy.append(sklearn.metrics.r2_score(train_data_y, forest.predict(train_data_x) ))

    # end tree estimator loop

        
    # fill accuracy plots
    plt.plot(estimator_list, test_accuracy, 'ro', label='test')
    plt.plot(estimator_list, train_accuracy, 'bo', label='train')
    #plt.plot(estimator_list, random_accuracy, 'yo', label='random')
    #plt.plot(estimator_list, most_freq_accuracy, 'go', label='most frequent')
    plt.title(title)
    plt.xlabel('Model Complexity')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('crossVal/test_train_error_'+title+'.pdf')
    


def confusion_Matrix(dataframe):
    
    '''
    Takes in pandas dataframe and plots the confusion matrix for multiClass classification
    '''

    print '-----> Creating Confusion Matrix...'

    plt.ion()
    
    # convert pandas to numpy array
    dataset = dataframe.values

    
    # Confusion Matrix
    forest = algorithm(n_estimators = 10,
                       max_features = None,
                       min_samples_split = 2,
                       max_depth = 10,
                       )

    y_pred = forest.fit(train_data_x, train_data_y).predict(test_data_x)
    
    cm = sklearn.metrics.confusion_matrix(test_data_y, y_pred)


    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title+'_Confusion Matrix')
    plt.savefig('crossVal/confusion_matrix_'+title+'.pdf')




def train_optimizer(dataframe):
    
    '''
    Takes in pandas dataframe and uses scikit to perform several cross validation tests. PLots and results are printed
    See: scikit.com
    '''
    
    print '\n\n============ Optimiziing Search Space =============\n'
    
    plt.ion()
    
    # convert pandas to numpy array
    dataset = dataframe.values

    # split training set into a test and train part.  80% train, 20% Test
    train_data, test_data = cross_validation.train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Now split the data into x(features) and y(known outcomes)
    train_data_x = train_data[:, 1:]
    train_data_y = train_data[:, 0]
    
    test_data_x = test_data[:, 1:]
    test_data_y = test_data[:, 0]

    # define the parameter search space
    search_space = [
        { 'max_features': ['auto', 'log2', 'sqrt'],
          'min_samples_split': [1, 2, 3, 4],
          'max_depth': [1, 3, 5],
          'learning_rate': [0.1, 0.5, 1],
         # 'min_samples_leaf': [1, 2, 3, 4, 5],
         # 'max_leaf_nodes': [1, 2, 3, 4, 5]
          }
        ]

    # Define the type of algrothm to employ
    #model = RandomForestClassifier(n_estimators = 10, n_jobs=-1)
    model = GradientBoostingClassifier(n_estimators = 10)
    
    print '=== Tuning hyper-parameters ==='
    
    cv = GridSearchCV(
        model, 
        search_space, 
        scoring='accuracy')
    
    cv.fit(train_data_x, train_data_y)
    
    print '\n Best parameters found on training set:', cv.best_estimator_
    
    print '\n\n Grid scores on training set:'
    for params, mean_score, scores in cv.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

    # if performing classification print a report out
    print("\n\nDetailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    print classification_report(test_data_y, cv.predict(test_data_x))



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

    '''  Takes in a numpy array of features and computes covariance
    '''

    print '\n=============== Covariance =======================\n'
    
    emp_cov = (np.dot(data_x.T, data_x)).astype(np.float32)
    
    vmax = emp_cov.max()
    plt.imshow(emp_cov, interpolation='nearest', #vmin=-vmax, vmax=vmax, 
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('Correlation')
    plt.savefig('features/manual_correlation.png')
    plt.clf()
    
    new_cov = sklearn.covariance.empirical_covariance(data_x)
    
    plt.imshow(new_cov, interpolation='nearest', #vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('Sci-Kit Covariance')
    plt.savefig('features/scikit_covar.png')
    plt.clf()

    print '\n============ End Covariance ===================\n'




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


def mutual_info(train, target):

    '''
    Takes in training dataframe of features

    '''

    print '-----> Calculating Mutual Information...'

    '''
    # initialize MI visiual matrix 
    n = train.shape[1]
    mi = np.zeros((n, n))
        
    for x, colx in enumerate(train):
    
        for y, coly in enumerate(train):

            MI = sklearn.metrics.mutual_info_score(train[colx], train[coly])
            
            #store value in matrix
            mi[x][y] = MI
    
    # end column loops
            
    # Plot the MI
    plt.imshow(mi, interpolation='nearest')
    plt.savefig('features/mutual_info_features.pdf')
    '''

    # Now plot feature vs target
    y_label = np.arange(len(list(train.columns)))
    mi = []

    for col in train:

        mi.append(sklearn.metrics.mutual_info_score(train[col], target))
        
    # Fill bar plot
    plt.barh(y_label, mi, align='center', color='red')
    plt.yticks(y_label, list(train.columns))
    #plt.autoscale(True)
    plt.tight_layout()

    plt.savefig('features/mutual_info_target.pdf')
    
    







