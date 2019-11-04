#for multi class classification

import random
import numpy as np
import tensorflow as tf
import math, sys
import logging
from preprocess import preprocess as pp
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import scipy.io as sio  # The library to deal with .mat

# Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 2
n_output = 2
# Learning parameters
learning_constant = 0.2
number_epochs = 1000
batch_size = 1000
crossValidationK = 10

# Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# DEFINING WEIGHTS AND BIASES

# Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
# Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
# Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))

# Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
# Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
# Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))


def multilayer_perceptron(input_d):
    # Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    # Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    # Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_2, w3), b3)

    return out_layer


# Create model
neural_network = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))

#Switch to softmax
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

#Init class names
class_names = ['AU06', 'AU10', 'AU12', 'AU14', 'AU17']

#STEP1 TODO: load bd4d data
batch_x, batch_y = pp().run()

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# RANDOMISE
shuffled_x, shuffled_y = unison_shuffled_copies(batch_x, batch_y)

print (shuffled_x.shape, shuffled_y.shape)
sys.exit(0)
# SPLITTING
split_x = np.array_split(batch_x, crossValidationK)
split_y = np.array_split(batch_y, crossValidationK)

mean_acc = 0

for i in range(crossValidationK):
    batch_x_train = np.arange(2).reshape(1, 2)
    batch_y_train = np.arange(2).reshape(1, 2)

    # ASSIGN
    batch_x_test = split_x[i]
    batch_y_test = split_y[i]

    for j in range(crossValidationK):
        if i != j:
            batch_x_train = np.append(batch_x_train, split_x[j], axis=0)
            batch_y_train = np.append(batch_y_train, split_y[j], axis=0)

    with tf.Session() as sess:
        sess.run(init)
        # Training epoch
        for epoch in range(number_epochs):

            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
            # Display the epoch
            if epoch % 100 == 0:
                print("Epoch:", '%d' % epoch)

        # Test model
        pred = neural_network  # Apply softmax to logits
        accuracy = tf.keras.losses.MSE(pred, Y)
        print("Accuracy:", accuracy.eval({X: batch_x_train, Y: batch_y_train}))
        # tf.keras.evaluate(pred,batch_x)

        print("Prediction:", pred.eval({X: batch_x_train}))
        output = neural_network.eval({X: batch_x_train})


        plt.plot(batch_y_train[0:10], 'ro', output[0:10], 'bo')
        plt.ylabel('some numbers')
        #plt.show()

        estimated_class = tf.argmax(pred, 1)  # +1e-50-1e-50
        correct_prediction1 = tf.equal(tf.argmax(pred, 1), label)
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

        print(accuracy1.eval({X: batch_x}))
        mean_acc += accuracy1.eval({X: batch_x})

mean_acc /= crossValidationK
print("Mean Accuracy: ", mean_acc)
