import tensorflow as tf
from math import sqrt
from utils import get

def normalize(X):
    """
    Normalizes the tensor X to have mean 0 and variance 1
    """
    mean = tf.reduce_mean(X)
    std = tf.sqrt(tf.reduce_mean(tf.square(X - mean)))
    return (X - mean) / std

def shrink(X, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1]):
    """
    Resizes the image tensor by averaging the pixels within the kernel window
    """
    return tf.nn.avg_pool(X, ksize=kernel, strides=stride, padding='SAME')
    
def grow(X, input_width, channels, scale, crop_size):
    """
    Upsamples tensor X via a learned neural deconvolution operation, then crops
    each upsampled image to shape (crop_size, crop_size, 3)
    """
    magnified_length = input_width * scale
    crop_offset = (magnified_length - crop_size) // 2
    W = tf.Variable(tf.random_normal([5, 5, 3, channels], stddev=0.01))
    b = tf.Variable(tf.constant(0.00, shape=[3]))
    X = tf.reshape(X, [-1, input_width, input_width, channels])
    upsample_shape = [tf.shape(X)[0], magnified_length, magnified_length, 3]
    conv = tf.nn.conv2d_transpose(
        X, W, upsample_shape, strides=[1, scale, scale, 1])
    conv = tf.nn.bias_add(conv, b)
    crop = tf.slice(
        conv,
        [0, crop_offset, crop_offset, 0],
        [tf.shape(conv)[0], crop_size, crop_size, 3])
return crop

def fully_connected_layer(input, num_input, num_output, weight_init):
    weights = tf.Variable(tf.truncated_normal([num_input, num_output],
     stddev=weight_init), name="weights")
    bias = tf.Variable(tf.constant(0.01, shape=[num_output]), name="bias")
    layer_output = tf.matmul(input, weights) + bias
    layer_output = tf.nn.elu(layer_output)
    return layer_output

def autoencoder(orig):
    """
    Constructs a neural autoencoder consisting of a learned, non-linear
    downsampling and upsampling. Returns the compressed image embedding
    and the reconstructed image.
    """
    #Compression
    pooled = shrink(orig)
    flattened = tf.reshape(pooled, [-1, 16*16*3])
    compressed = fully_connected_layer(flattened, 768, 128, sqrt(0.1*0.1/768))
    compressed = fully_connected_layer(compressed, 128, 64, sqrt(0.1*0.1/128))
    #Reconstruction
    reconstructed = fully_connected_layer(compressed, 64, 20736, sqrt(0.1*0.1/64))
    repr_dim = get('autoencoder.ae_repr_dim')
    reconstructed = grow(reconstructed, input_width=18, channels=repr_dim,
     scale=2, crop_size=32)
    reconstructed = normalize(reconstructed)
    return compressed, reconstructed

def naive(orig):
    """
    Performs downsampling and upsampling on the tensor X via linear operations
    """
    scale = get('autoencoder.naive_scale')
    compressed = shrink(
        orig, kernel=[1, scale, scale, 1], stride=[1, scale, scale, 1])
    reconstructed = normalize(compressed)
    return compressed, reconstructed
