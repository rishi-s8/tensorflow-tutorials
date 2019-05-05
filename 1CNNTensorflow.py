
# coding: utf-8
'''
Author: Rishi Sharma
'''

import tensorflow as tf
import numpy as np
from keras.utils import np_utils

#Load Dataset
x_train = np.load('./data/mnist/x_train.npy')
x_test = np.load('./data/mnist/x_test.npy')
y_train_cls = np.load('./data/mnist/y_train.npy')
y_test_cls = np.load('./data/mnist/y_test.npy')

#Define variables
num_pixels = x_train.shape[1]
num_channels = 1

#TypeCast
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize
x_train = x_train/255
x_test = x_test/255

#One-hot encoding
y_train = np_utils.to_categorical(y_train_cls)
y_test = np_utils.to_categorical(y_test_cls)
num_classes = y_train[0].size

#Filter Constants
filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 36

fc_size = 128

#Define Placeholders
x = tf.placeholder(tf.float32, [None, num_pixels, num_pixels])
x_img = tf.reshape(x,[-1,num_pixels,num_pixels,num_channels])

y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

#Helper Functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input = input, filter = weights, strides = [1,1,1,1], padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    layer = tf.nn.relu(layer)
    print("No. of parameters: " +  str(layer.get_shape()[1:4].num_elements()))
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1,num_features])
    return layer_flat, num_features

def new_fc_layer(input,num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer=tf.matmul(input,weights)+biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

#First Conv Layer
layer_conv1, weights_conv1 = new_conv_layer(input=x_img, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)

#Print the layer_conv1
layer_conv1

#Second Conv Layer
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

#Print the layer_conv2
layer_conv2

#Add Flattening layer
layer_flat, num_features = flatten_layer(layer_conv2)

#Print the layer_flat
layer_flat

#Add FC Layers
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

layer_fc2

#Prediction
y_pred = tf.nn.softmax(layer_fc2)
#Prediction Class
y_pred_cls = tf.argmax(y_pred, axis=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)

cost = tf.reduce_mean(cross_entropy)


lr = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

#No. of correct predictions
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

#Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Initialize session
session=tf.Session()

#initialize variables
session.run(tf.global_variables_initializer())


def print_accuracy():
    acc = session.run(accuracy, feed_dict = feed_dict_test)
    print("Accuracy on test-set:{0:.1%}".format(acc))


#BatchSize
batch_size = 64

def optimize(num_iterations):
    for j in range(num_iterations):
        print("Iteration: " + str(j))
        for i in range(batch_size, x_train.shape[0], batch_size):
            x_batch = x_train[i-batch_size:i]
            y_true_batch = y_train[i-batch_size:i]
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        print_accuracy()

feed_dict_test = {x:x_test, y_true: y_test, y_true_cls: y_test_cls}

print_accuracy()

weights_conv1.eval(session=session)

weights_conv2.eval(session=session)

optimize(num_iterations=2)

weights_conv1.eval(session=session)

weights_conv2.eval(session=session)
