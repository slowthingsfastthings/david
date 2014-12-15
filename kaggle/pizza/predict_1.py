######################
# Intro code for Kaggle website Titanic project
#
# Introduces numpy, pandas, and scikit
#
# by David Curry, 8/10/2014
#
#####################

print '-----> Importing Modules'

<<<<<<< HEAD
<<<<<<< HEAD

#changes!!

=======
>>>>>>> 0731d796a4da7f984bd12cb926e99cb107d5dbcd
=======
>>>>>>> 0731d796a4da7f984bd12cb926e99cb107d5dbcd
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics


print '-----> Importing Data'

train = pd.io.json.read_json('train.json')
test  = pd.io.json.read_json('test.json') 


# change recived pizza from True/False to 0 and 1.  Always put known answers at the first column
train['recieved'] = train['requester_received_pizza'].map( {False:0, True:1 } ).astype(int)
cols = train.columns.tolist()           # get column list
train = train[ cols[-1:] + cols[:-1] ]  # put last coumn in front


# Drop columns from test data that we dont use.  Keep test ids for web submission
ids = test['request_id'].values

test = test.drop(['giver_username_if_known', 'request_text_edit_aware', 'request_title', \
                       'requester_username', 'unix_timestamp_of_request_utc', 'requester_subreddits_at_request', 'request_id', \
                        ], axis=1)


# or alternately, force train to have same columns as test, since test is much smaller.
test_cols = test.columns.tolist()
#print test_cols

train = train[ ['recieved', u'requester_account_age_in_days_at_request', u'requester_days_since_first_post_on_raop_at_request', u'requester_number_of_comments_at_request', u'requester_number_of_comments_in_raop_at_request', u'requester_number_of_posts_at_request', u'requester_number_of_posts_on_raop_at_request', u'requester_number_of_subreddits_at_request', u'requester_upvotes_minus_downvotes_at_request', u'requester_upvotes_plus_downvotes_at_request', u'unix_timestamp_of_request'] ]




#print test.info()
#print '\n\n', train.info()

# Whats empty?
for col in train:
    if len( train[train[col].isnull()] ) > 0: print col

for col in test:
    if len( test[test[col].isnull()] ) > 0: print col

# convert back to numpy array type.  Scikit will require this.
train_data = train.values
test_data  = test.values




# ================================================================================
# Datasets are ready for SciKit Machine Learning!

Cprint '-----> Training...'

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

output = forest.predict(test_data).astype(int)

print '-----> Predicting...'

#print forest.score(train_data[0::,1::], forest.predict(train_data[0::,1::]).astype(int))

print metrics.classification_report(train_data[0::,0], forest.predict(train_data[0::,1::]) )


# Save into Kaggle submission format
predictions_file = open("pizza_results.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["request_id","requester_received_pizza"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print '-----> Done.'


