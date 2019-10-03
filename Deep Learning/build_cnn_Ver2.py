import numpy as np
import tensorflow as tf
from math import sqrt
from utils import get

def fully_connected_layer(input, num_input, num_output, activation, weight_init):
    weights = tf.Variable(tf.truncated_normal([num_input, num_output],
     stddev=weight_init), name="weights")
    bias = tf.Variable(tf.constant(0.01, shape=[num_output]), name="bias")
    layer_output = tf.matmul(input, weights) + bias
    if activation == 'On':
        layer_output = tf.nn.leaky_relu(layer_output)
    return layer_output

def convolutional_layer(input, input_depth, filter_size, num_filter, pad,
 weight_init, stride_size):
    stride = [1, stride_size, stride_size, 1]
    weights = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size,
    input_depth, num_filter], stddev=weight_init), name="weights")
    bias = tf.Variable(tf.constant(0.01, shape=[num_filter]), name="bias")
    convolution = tf.nn.conv2d(input, weights, stride, pad)
    layer_output = tf.nn.leaky_relu(convolution + bias)
    return layer_output

def challenge(X, train):
    keep_prob = 0.5
    layer1_output = convolutional_layer(X, 3, 5, 64, 'SAME', sqrt(3.0/
     (32*32*3+32*32*64)), 1)
    layer2_output = convolutional_layer(layer1_output, 64, 5, 64, 'SAME',
     sqrt(3.0/(32*32*64+32*32*64)), 1)
    pooled = tf.nn.max_pool(layer2_output, ksize=[1,2,2,1], strides=[1,2,2,1],
     padding='SAME')
    layer3_output = convolutional_layer(pooled, 64, 5, 128, 'SAME', sqrt(3.0/
     (16*16*64+16*16*128)), 1)
    layer4_output = convolutional_layer(layer3_output, 128, 5, 128, 'SAME',
     sqrt(3.0/(16*16*128+16*16*128)), 1)
    pooled = tf.nn.max_pool(layer4_output, ksize=[1,2,2,1], strides=[1,2,2,1],
     padding='SAME')
    layer5_output = convolutional_layer(pooled, 128, 5, 256, 'SAME', sqrt(3.0/
     (8*8*128+8*8*256)), 1)
    layer6_output = convolutional_layer(layer5_output, 256, 5, 256, 'SAME',
     sqrt(3.0/(8*8*256+8*8*256)), 1)
    layer7_output = convolutional_layer(layer6_output, 256, 5, 256, 'SAME',
     sqrt(3.0/(8*8*256+8*8*256)), 1)
    pooled = tf.nn.max_pool(layer7_output, ksize=[1,2,2,1], strides=[1,2,2,1],
     padding='SAME')
    layer8_output = convolutional_layer(pooled, 256, 5, 512, 'SAME', sqrt(3.0/
        (4*4*256+4*4*512)), 1)
    layer9_output = convolutional_layer(layer8_output, 512, 5, 512, 'SAME',
        sqrt(3.0/(4*4*512+4*4*512)), 1)
    layer10_output = convolutional_layer(layer9_output, 512, 5, 512, 'SAME',
        sqrt(3.0/(4*4*512+4*4*512)), 1)
    flattened = tf.reshape(layer10_output, [-1, 4*4*512])
    layer12_output = fully_connected_layer(flattened, 4*4*512, 4*256, 'On',
        sqrt(3.0/(4*4*512+4*256)))

    if train == True:
        layer13_output = tf.nn.dropout(layer13_output, keep_prob)
        layer14_output = fully_connected_layer(layer13_output, 256, 64, 'On',sqrt(3.0/(256+64)))
        layer15_output = fully_connected_layer(layer14_output, 64, 32, 'On', sqrt(3.0/(64+32)))
        layer16_output = fully_connected_layer(layer15_output, 32, 4, 'NONE',sqrt(3.0/(32+4)))

    return layer16_output