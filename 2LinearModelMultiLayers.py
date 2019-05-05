
# coding: utf-8
'''
Author: Rishi Sharma
'''

import tensorflow as tf
import numpy as np
from keras.utils import np_utils

#Load Dataset
x_train = np.load('data/mnist/x_train.npy')
x_test = np.load('data/mnist/x_test.npy')
y_train_cls = np.load('data/mnist/y_train.npy')
y_test_cls = np.load('data/mnist/y_test.npy')

#Total Number of pixels
num_pixels = x_train.shape[1]*x_train.shape[2]

#Flatten images
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Normalize
x_train = x_train/255
x_test = x_test/255

#One-hot encoding
y_train = np_utils.to_categorical(y_train_cls)
y_test = np_utils.to_categorical(y_test_cls)
num_classes = y_train[0].size

#Placeholders
x = tf.placeholder(tf.float32, [None, num_pixels])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

#Placeholder for Learning Rate
step = tf.placeholder(tf.int32)

#Variables
K = 100
L = 100
M = 16
W1 = tf.Variable(tf.truncated_normal([num_pixels, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.zeros([M, num_classes]))
B4 = tf.Variable(tf.zeros([num_classes]))

#Define Graph
Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Ylogits = tf.matmul(Y3, W4) + B4

#Prediction
y_pred = tf.nn.softmax(Ylogits)
y_pred_cls = tf.argmax(y_pred, axis=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

# Define Optimizer
#the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/np.e)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

#No. of correct predictions
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Initialize Session
session=tf.Session()

#Initialize Variables
session.run(tf.global_variables_initializer())

#Test Function
def print_accuracy():
    acc = session.run(accuracy, feed_dict = feed_dict_test)
    print("Accuracy on test-set:{0:.1%}".format(acc))


#Optimizer function
batch_size = 1000
def optimize(num_iterations):
    for j in range(num_iterations):
        #Sending images in batches
        for i in range(batch_size, x_train.shape[0], batch_size):
            x_batch = x_train[i-batch_size:i]
            y_true_batch = y_train[i-batch_size:i]
            feed_dict_train = {x: x_batch, y_true: y_true_batch, step: j}
            session.run(optimizer, feed_dict=feed_dict_train)
        print_accuracy()

feed_dict_test = {x:x_test, y_true: y_test, y_true_cls: y_test_cls}

#Test without optimization
print_accuracy()


optimize(num_iterations=50)

#Print Weight matrix1
W1.eval(session=session)

#Print Biases 1
B1.eval(session=session)
