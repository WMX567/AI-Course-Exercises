import tensorflow as tf
from utils import get
import numpy as np
def supervised_placeholders():
    """
    Constructs the tensorflow placeholders needed as input to the network.
    Returns:
        two tensorflow placeholders. The first return value should be
        the placeholder for the image data. The second should be for the
        class labels.
"""
    image_dim = get('image_dim')
    images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 3])
    labels = tf.placeholder(tf.int64, [None,])
    return images, labels

def supervised_optimizer(loss):
    """
    Given the network loss, constructs the training op needed to train the
    network.
    Returns:
        the operation that begins the backpropogation through the network
        (i.e., the operation that minimizes the loss function).
"""
    learning_rate = get('cnn.learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return train_op

def unsupervised_placeholders():
    """
    Constructs the tensorflow placeholders needed as input to the autoencoder
    model.
    Returns:
        a tensorflow placeholder for the image data.
"""
    image_dim = get('image_dim')
    images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 3])
    return images

def unsupervised_optimizer(loss):
    """
    Constructs the training op needed to train the autoencoder model.
    Returns:
        the operation that begins the backpropogation through the network
        (i.e., the operation that minimizes the loss function).
"""
    learning_rate = get('autoencoder.classifier.learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return train_op

def challenge_placeholders():
    """
    Constructs the tensorflow placeholders needed as input to the network.
    Returns:
        at least two tensorflow placeholders. The first return value should
        be the placeholder for the image data. The second should be for the
        class labels. Note that depending on your choice of model you may
        need to add more placeholders.
"""
    image_dim = get('image_dim')
    images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 3])
    labels = tf.placeholder(tf.int64, [None,])
    train = tf.placeholder_with_default(False, [])
    return images, labels, train

def challenge_optimizer(loss):
    """
    Constructs the training op needed to train the challenge model. Depending
    on your choice of model, this may be the same as supervised_optimizer.
    Returns:
        the operation that begins the backpropogation through the network
        (i.e., the operation that minimizes the loss function).
"""
    learning_rate = get('autoencoder.learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return train_op

def cross_entropy_loss(labels, logits):
    """
    Given the ground truth labels and the logits from the output of the
    network, constructs the scalar cross entropy loss.
    Returns:
        the output of the cross entropy loss function as a Tensorflow Tensor
"""
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=logits))
    return loss

def predictions(logits):
    """
    Given the network output, determines the predicted class index
    Returns:
        the predicted class output as a Tensorflow Tensor
"""
    pred = tf.argmax(logits, axis=1)
    return pred

def accuracy(labels, logits):
    """
    Constructs the accuracy metric given the ground truth labels and the
    network output logits.
    Returns:
the accuracy value as a Tensorflow Tensor

"""
    pred_correct= tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))
    return accuracy

def mean_squared_error(images, reconstructed):
    """
    Constructs the mean squared error loss between the original images and the
    autoencoder reconstruction
    Returns:
        the mse loss as a Tensorflow Tensor
"""
    mse = tf.losses.mean_squared_error(images, reconstructed)
    return mse
    
def get_class_examples(food, label):
    """
    @in: food - instance of FoodDataset
    @in: label - the label that you want to pull the validation subset for
    Returns the validation examples and labels for a particular label
    """
    # TODO: Use this function for 3i.
    valid = food.get_examples_by_label(partition='valid', label=label)
    labels = np.ones(valid.shape[0]) * label
    return valid, labels