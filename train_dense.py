'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Aniket Suri
Roll No.: 14CS10004

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import tensorflow as tf
import numpy as np
import os
    
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
n_classes = 10

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    trainX = np.reshape(trainX,(trainX.size/784,784))
    trainX = np.matrix(trainX)
        
    trainY = np.matrix(trainY).T
    trainY_ = np.zeros((trainX[:,0].size,n_classes))
    
    print "trainY_: ",trainY_[:,0].shape," ",trainY_[0,:].shape
    for i in range(trainY[:,0].size):
        trainY_[i,trainY[i]] = 1
    trainY = trainY_

    # Layer's sizes
    x_size = trainX.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = trainY.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, 1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        hm_epochs = 20
        batch_size = 100
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for k in xrange(0,trainX[:,0].size,batch_size):
                epoch_x,epoch_y = trainX[k:k+batch_size,:], trainY[k:k+batch_size,:]

                sess.run(updates, feed_dict={X: epoch_x, y: epoch_y})

            train_accuracy = np.mean(np.argmax(trainY, axis=1) ==
                                 sess.run(predict, feed_dict={X: trainX, y: trainY}))

            print("Epoch = %d, train accuracy = %.2f%% "  % (epoch + 1, 100. * train_accuracy))
    
        W0 = w_1.eval()
        W1 = w_2.eval()
        np.savez('weights_dense.npz', a=W0, b=W1)


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    testX = np.reshape(testX,(testX.size/784,784))
    testX = np.matrix(testX)
        

    # Layer's sizes
    x_size = testX.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, n_classes])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, n_classes))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, 1)
    
    with tf.Session() as sess:
        data = np.load('weights_dense.npz')
        W0 = data['a']
        W1 = data['b']
        assign_op = w_1.assign(W0)
        sess.run(assign_op)
        assign_op = w_2.assign(W1)
        sess.run(assign_op)
        labels = sess.run(predict, feed_dict={X: testX})
        
    return labels
