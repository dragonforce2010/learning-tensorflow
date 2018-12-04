# -*- coding: utf-8 -*-
import os
# silence warning: tensorflow using SSE4.1, SSE4.2, and AVX
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

# create input data
X = np.linspace(-7, 7, 180) 

# implementation of activation functions
def sigmoid(X):
    Y = [ 1 / (1 + np.exp(-x)) for x in X]
    return Y

def relu(X):
    Y = [ x * (x > 0) for x in X]
    return Y

def tanh(X):
    Y = [ (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-1)) for x in X]
    return Y

def softplus(X):
    Y = [ np.log(1 + np.exp(x)) for x in X]
    return Y

# Use tensorflow buildin functions instead of the ones we manually implementatied above
y_sigmoid = tf.nn.sigmoid(X)
y_relu = tf.nn.relu(X)
y_tanh = tf.nn.tanh(X)
y_softplus = tf.nn.softplus(X)

# create sessions
sess = tf.Session()
y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, 
                                                  y_relu, 
                                                  y_tanh, 
                                                  y_softplus])

# plot the graph
plt.subplot(221)
plt.plot(X, y_sigmoid, c="blue", label="Sigmoid")
plt.legend(loc='best')

plt.subplot(222)
plt.plot(X, y_relu, c='red', label="Relu")
plt.legend(loc='best')

plt.subplot(223)
plt.plot(X, y_tanh, c='green', label='Tanh')
plt.legend(loc='best')

plt.subplot(224)
plt.plot(X, y_softplus, c='yellow', label='Softplus')
plt.legend(loc='best')

# show the graph
plt.show()

sess.close()