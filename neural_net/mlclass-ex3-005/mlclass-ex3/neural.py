# Neural Network code for Stanford COurse
#
# By David Curry
# 4/27/2014
#
import time 
import scipy as sp
import scipy.io as sio
import numpy as np
import math

# We will need this
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

print '----> Loading dataset'
data = sio.loadmat('ex3data1.mat');
weights = sio.loadmat('ex3weights.mat')


# define the different arrays in loaded files
theta1 = weights['Theta1']
theta2 = weights['Theta2']
x = data['X']
y = data['y']

# add bias in row 1
x = np.c_[ np.ones(x.shape[0]), x ]

# the sigmoid functionw e will use
#sp.special.expit(x)

# We now need to propogate forward through the neural network.
# First, the hidden layer

z_2 = x.dot(theta1.T)

g_2 = sigmoid(z_2)

