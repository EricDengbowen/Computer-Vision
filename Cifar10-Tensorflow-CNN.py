
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import math


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data():
    
    train1=unpickle('cifar-10-batches-py/data_batch_1')
    train2=unpickle('cifar-10-batches-py/data_batch_2')
    train3=unpickle('cifar-10-batches-py/data_batch_3')
    train4=unpickle('cifar-10-batches-py/data_batch_4')
    train5=unpickle('cifar-10-batches-py/data_batch_5')
   

    X_train_orig=np.vstack((train1[b'data'],train2[b'data'],train3[b'data'],train4[b'data'],train5[b'data']))
    Y_train_orig=np.hstack((train1[b'labels'],train2[b'labels'],train3[b'labels'],train4[b'labels'],train5[b'labels']))
    
    X_train_change=X_train_orig.reshape(50000,3,32,32).transpose((0,2,3,1))
    
    test=unpickle('cifar-10-batches-py/test_batch')
    X_test_change=test[b'data'].reshape(10000,3,32,32).transpose((0,2,3,1))
    Y_test_change=np.array(test[b'labels'])
    
    dummy_y_train=tf.one_hot(Y_train_orig,10)
    dummy_y_test=tf.one_hot(Y_test_change,10)
    
   
    
    with tf.Session() as sess:
        
        final_train_y=sess.run(dummy_y_train)
        final_test_y=sess.run(dummy_y_test)
  
    
    return X_train_change,final_train_y,X_test_change, final_test_y


X_train,Y_train,X_test,Y_test=load_data()


def create_placeholders(n_H0,n_W0,n_C0,n_y):
    
    X=tf.placeholder(tf.float32,shape=(None,n_H0,n_W0,n_C0))
    Y=tf.placeholder(tf.float32,shape=(None,n_y))
    #n_y = # of labels
    
    return X,Y
    

def initialize_parameters():
    W1=tf.get_variable('W1',[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer())
    W2=tf.get_variable('W2',[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer())
    
    parameters={'W1':W1,
                'W2':W2}
    return parameters

def forward_propagation(X,parameters):
    W1=parameters['W1']
    W2=parameters['W2']
    
    Z1=tf.nn.conv2d(X,W1,strides=(1,1,1,1),padding="SAME")
    A1=tf.nn.relu(Z1)
    P1=tf.nn.max_pool(A1,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME")
    
    Z2=tf.nn.conv2d(P1,W2,strides=(1,1,1,1),padding="SAME")
    A2=tf.nn.relu(Z2)
    P2=tf.nn.max_pool(A2,ksize=(1,4,4,1),strides=(1,4,4,1),padding="SAME")
    P2=tf.contrib.layers.flatten(P2) 
    #tf.contrib.layers.flatten(P): given an input P, this function flattens each example into a 1D vector 
    #it while maintaining the batch-size. It returns a flattened tensor with shape [batch_size, k].
    Z3=tf.contrib.layers.fully_connected(P2,10,activation_fn=None)
    
    
    
    return Z3
    
    
def compute_cost(Z3,Y):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    
    return cost



def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:m,:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 1024, print_cost = True):
   
    
  
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # Initialize parameters
    parameters = initialize_parameters() 
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)  
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)   
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initialize all the variables globally
    init = tf.global_variables_initializer()     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                ### END CODE HERE ###
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters



_, _, parameters = model(X_train, Y_train, X_test, Y_test)

