#  A script to perform machine learning lagos on cms pt data
#
# By David Curry
#
#

import sys
import numpy as np
from sklearn import datasets, linear_model, cross_validation, svm
import pylab as pl

print '\n\n ======== You Are Now Running pt_ML.py .... Good Luck! =======\n\n'

# Set the print level. Default = 0
if len(sys.argv) is 1: printLevel = 0
else: printLevel = sys.argv[1]

#### Open data file ####
print '---> Importing the Dataset\n' 
data = np.loadtxt('test.txt')

#### Split dataset into training(70%) and test(30%) ####

#This part is dataset dependent. 
# Last coumn is dependent(y) variable.  Rest are features(x)
data_x = data[:, 0:data.shape[1]-1]
data_y = data[:, data.shape[1]-1]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(data_x, data_y, test_size=0.4, random_state=0)

print 'Data shape    = ', data.shape
print 'X Data shape  = ', data_x.shape  
print 'Y Data shape  = ', data_y.shape
print 'X train shape = ', x_train.shape
print 'Y train shape = ', y_train.shape


#  Creat list of data entries for post processing tests
num_entries = [100, 1000, 10000, x_train.shape[0] ]

regr_var  = []
ridge_var = []

for i in range( len(num_entries) ):
    
    print ' Testing on', num_entries[i], 'datapoints'

    x_train_t = x_train[:num_entries[i]]
    y_train_t = y_train[:num_entries[i]]

    # Create linear regression object
    regr = linear_model.LinearRegression().fit(x_train_t, y_train_t) 
    
    regr_var.append(regr.score(x_test, y_test))

    # Create Regularized Regression(Ridge technique) 
    ridge = linear_model.Ridge().fit(x_train_t, y_train_t) 

    ridge_var.append(ridge.score(x_test, y_test))


# Store the coefficients to file
#regr_coef = np.append(regr.intercept_, regr.coef_)
#np.save('coef', regr_coef)


# ===== Test coefficients ======
#Add a column of ones to X (interception data)
#it = np.ones(shape=(test_data_x.shape[0], 1))
#
#temp_x = np.append(it, test_data_x, 1)
#
#print 'Shape regr_coef = ', regr_coef.shape
#print 'Shape temp_x    = ', temp_x.shape
#
#ans = temp_x.dot(regr_coef)
#
#for i in range(10):
#    print 'Prediction = ', ans[i]
#    print 'True       = ', test_data_y[i]



# ==== Plot Various Results ====
pl.ion()
pl.plot(num_entries, regr_var)
pl.xscale('log')
pl.show()


#pl.ion()
#pl.hist( (regr.predict(x_test) - y_test)/y_test , bins = 50, range = (-3,3) )
#pl.show()

_ = raw_input("Press [enter] to continue.")
pl.close()


