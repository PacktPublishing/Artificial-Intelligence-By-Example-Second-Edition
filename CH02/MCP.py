# McCulloch Pitt Neuron built with Tensorflow and represented with
# Tensorflow 2.0.0-beta1
# Copyright 2019 Denis Rothman MIT License. See LICENSE.

import tensorflow as tf
import numpy as np
import math

print(tf.__version__)

# The variables
x = tf.Variable([[0.0,0.0,0.0,0.0,0.0]], dtype = tf.float32)
W = tf.Variable([[0.0],[0.0],[0.0],[0.0],[0.0]], dtype = tf.float32)
b = tf.Variable([[0.0]])

# The Neuron
def neuron(x, W, b):
    y1=np.multiply(x,W)+b
    y1=np.sum(y1)
    y = 1 / (1 + np.exp(-y1)) #logistic Sigmoid 
    return y  

# The data
x_1 = [[10, 2, 1., 6., 2.]]
w_t = [[.1, .7, .75, .60, .20]]
b_1 = [1.0]

# Computing the value of the neuron
value=neuron(x_1,w_t,b_1)

# Availability of the location computed
availability=1-value
print("value for threshold calculation:{0:.5f}".format(round(value,5)))
print("Availability of location x     :{0:.5f}".format(round(availability,5)))

''' Output:
value for threshold calculation:0.99999
Availability of location x     :0.00001
'''


