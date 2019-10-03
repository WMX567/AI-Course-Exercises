import numpy as np
import tensorflow as tf
from math import sqrt
from utils import get

def cnn(X):
    layer1_output = convolutional_layer(X, 3, 5, 64, 'SAME', np.sqrt(1/25/3), 2)
    layer2_output = convolutional_layer(layer1_output, 64, 5, 32, 'SAME',
     np.sqrt(1/25/64), 2)
    layer3_output = convolutional_layer(layer2_output, 32, 5, 16, 'SAME',
     np.sqrt(1/25/32), 2)
    #Output from layer3 is 4*4*64
    flattened = tf.reshape(layer3_output, [-1, 4*4*16])
    layer4_output = fully_connected_layer(flattened, 4*4*16, 64, 'ReLU',
     np.sqrt(1/256))
    layer5_output = fully_connected_layer(layer4_output, 64, 32, 'ReLU',
     np.sqrt(1/64))
    layer6_output = fully_connected_layer(layer5_output, 32, 4, 'NONE',
     np.sqrt(1/32))
    return layer6_output

def convolutional_layer(input, input_depth, filter_size, num_filter, pad,
 weight_init, stride_size):
    #Initialize weight [filter_height, filter_width, in_depth, out_depth]
    #Intialize bias = number of filter
    stride = [1, stride_size, stride_size, 1]
    weights = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size,
    input_depth, num_filter], stddev=weight_init), name="weights")
    bias = tf.Variable(tf.zeros(num_filter), name="bias")
    convolution = tf.nn.conv2d(input, weights, stride, pad)
    layer_output = tf.nn.relu(convolution + bias)
    return layer_output
    
def fully_connected_layer(input, num_input, num_output, activation, weight_init):
    weights = tf.Variable(tf.truncated_normal([num_input, num_output],
     stddev=weight_init), name="weights")
    bias = tf.Variable(tf.zeros(num_output), name="bias")
    layer_output = tf.matmul(input, weights) + bias
    if activation == 'ReLU':
        layer_output = tf.nn.relu(layer_output)
    return layer_output