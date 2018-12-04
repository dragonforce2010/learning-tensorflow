import tensorflow as tf

# download the mnist data
from tensorflow.examples.tutorials.mnst import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

