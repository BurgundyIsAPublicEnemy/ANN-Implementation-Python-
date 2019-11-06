import numpy as np
import tensorflow as tf
import math, random
import logging, sys
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn import metrics

import pandas as pd

x = pd.read_csv("predx_for_regression.csv", header=0)
x_array = np.asarray(x, dtype = "float")
x_arrayt=x_array

y = pd.read_csv("predy_for_regression.csv", header=0)
y_array = np.asarray(y, dtype = "float")
y_arrayt=y_array

whole_data=np.concatenate((x_arrayt, y_arrayt),axis=1)


angle = pd.read_csv("angle.csv", header=0)
angle_array = np.asarray(angle, dtype = "float")
angle_arrayt=angle_array.transpose()


#Network parameters
n_hidden1 = 147
n_hidden2 = 75
n_hidden3 = 50
n_input = 98
n_output = 1
n_slices = 10
#Learning parameters
learning_constant = 0.02
number_epochs = 10
batch_size = 20000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES

#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
#Biases output layer
b3 = tf.Variable(tf.random_normal([n_hidden3]))

b4 = tf.Variable(tf.random_normal([n_output]))


#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
#Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))

w4 = tf.Variable(tf.random_normal([n_hidden3, n_output]))



#The incoming data given to the
#network is input_d
def multilayer_perceptron(input_d):
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(input_d, w1), b1))
    #Task of neurons of second hidden layer
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, w2), b2))
    #Task of neurons of output layer
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3), b3))
    #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_3, w4),b4)

    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))

#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()
#Create a session

#Initialize and prepare data sets
batch_x=(whole_data-200)/2000
temp=np.array([angle_array[:,0]])
batch_y=temp.transpose()

#K-FOLD PREPERATION
#Randomise data set to reduce overfitting / generalise the data
zipped = list(zip(batch_x, batch_y))
random.shuffle(zipped)
batch_x, batch_y = zip(*zipped)

#Set up the K-Fold Cross Validation
kf = KFold(n_slices)

#Slice batches into n_slices
batch_x = np.array_split(batch_x, n_slices)
batch_y = np.array_split(batch_y, n_slices)
angle_array = np.array_split(batch_y, n_slices)

#Initialize mean accuracy variables for reporting
mean_acc = 0
mean_per_acc = 0

#Generate arrays of indexes based on number of slices in the K fold
for train_index, test_index in kf.split(batch_x):
    #Show which indexes we are using
    print(train_index, test_index)

    #Setting test labels
    label=angle_array#+1e-50-1e-50

    #Setting test batches
    batch_x_test=batch_x[int(test_index)]
    batch_y_test=batch_y[int(test_index)]

    #Setting training batches by iterating over the training arrays
    for j in train_index:
        #if training batches aren't init. build them
        try:
            batch_x_train = np.append(batch_x_train, batch_x[j], axis=0)
            batch_y_train = np.append(batch_y_train, batch_y[j], axis=0)
        except:
            batch_x_train = batch_x[j]
            batch_y_train = batch_y[j]

    #Set label arrays
    label_train=label
    label_test=label

    with tf.Session() as sess:
        sess.run(init)
        #Training epoch
        for epoch in range(number_epochs):
            #Run the optimizer feeding the network with the batch
            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
            #Display the epoch
            if epoch % 100 == 0 and epoch>10:
                print("Epoch:", '%d' % (epoch))
                print("Accuracy:", loss_op.eval({X: batch_x_train, Y: batch_y_train}) )



        # Test model
        pred = (neural_network)  # Apply softmax to logits
        accuracy=tf.keras.losses.MSE(pred,Y)
        ac1 = np.square(accuracy.eval({X: batch_x_test, Y: batch_y_test})).mean()
        #tf.keras.evaluate(pred,batch_x)

        #Plot predictions
        output=neural_network.eval({X: batch_x_test})
        plt.plot(batch_y_test, 'r', output, 'b')
        plt.ylabel('some numbers')
        #plt.show()

        #Second plot
        plt.plot(batch_y_train[30000:300020], 'r', output[30000:300020], 'b')
        plt.ylabel('some numbers')
        #plt.show()

        #Debug: output the output of model
        # print(batch_y_train[30000:300020])
        # print(output[30000:300020])
        # df = DataFrame(output)
        # export_csv = df.to_csv ('output.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
        # print (df)

        #Sum for mean accuracy
        mean_acc += ac1
        #Reset batches for data
        batch_x_train = []
        batch_y_train = []

#Calculate mean averages
mean_acc = mean_acc / n_slices

#Report back accuracy and HYPERPARAMS
try:
    print('Mean accuracy: ', mean_acc)
    f = open("reportlog.txt", "a")
    f.write('\n\n\n')
    f.write(("HYPERPARAMS: " + " \n HIDDEN LAYER 1 NO. NODES: " + str(n_hidden1) + "\n HIDDEN LAYER 2 NO. NODES: " + str(n_hidden2) + "\n HIDDEN LAYER 3 NO. NODES: " + str(n_hidden3) + "\n INPUT NODES: " +  str(n_input) + "\n OUTPUT NODES: " + str(n_output) +  "\n NSLICES: " + str(n_slices)))
    f.write(("\n HYPERPARAMS 2: \n LEARNING_CONSTANT: " + str(learning_constant) + "\n NO EPOCHS: " + str(number_epochs) + "\n BATCHSIZE: " + str(batch_size)))
    f.write(("\n MEAN MSE ACC: " + str(mean_acc)))
    f.write('\n\n\n')
    f.close()
except Exception as e:
    print(e)
    print('Something went wrong here so Mean accuracy: ', mean_acc)
