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

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 300
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables

init = tf.initialize_all_variables()




def train(trainX, trainY):
    '''
    Complete this function.
    '''
    with tf.Session() as sess:
        trainX = np.reshape(trainX,(trainX.size/784,-1))
        trainX = np.matrix(trainX)

        
        trainY = np.matrix(trainY).T
        trainY_ = np.zeros((trainX[:,0].size,n_classes))
        
        for i in range(trainY[:,0].size):
            trainY_[i,trainY[i]] = 1
        trainY = trainY_

        sess.run(init)
        hm_epochs = 5
        for epoch in range(hm_epochs):

            print("Hello")
            for k in xrange(0,trainX[:,0].size,batch_size):
                batch_x,batch_y = trainX[k:k+batch_size,:], trainY[k:k+batch_size,:]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y,keep_prob: 1.})
                print("Iter ",k,", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        print("Optimization Finished!")

        wc1 = weights['wc1'].eval()
        wc2 = weights['wc2'].eval()
        wd1 = weights['wd1'].eval()
        out_c = weights['out'].eval()
        bc1 = biases['bc1'].eval()
        bc2 = biases['bc2'].eval()
        bd1 = biases['bd1'].eval()
        out_b = biases['out'].eval()
        np.savez('weights.npz', a=wc1, b=wc2, c=wd1, d=out_c, e=bc1, f=bc2, g=bd1, h=out_b)


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
    testX = np.reshape(testX,(testX.size/784,-1))
    testX = np.matrix(testX)

    with tf.Session() as sess:
        data = np.load('weights.npz')
        wc1 = data['a']
        assign_op = weights['wc1'].assign(wc1)
        sess.run(assign_op)
        wc2 = data['b']
        assign_op = weights['wc2'].assign(wc2)
        sess.run(assign_op)
        wd1 = data['c']
        assign_op = weights['wd1'].assign(wd1)
        sess.run(assign_op)
        out_c = data['d']
        assign_op = weights['out'].assign(out_c)
        sess.run(assign_op)
        bc1 = data['e']
        assign_op = biases['bc1'].assign(bc1)
        sess.run(assign_op)
        bc2 = data['f']
        assign_op = biases['bc2'].assign(bc2)
        sess.run(assign_op)
        bd1 = data['g']
        assign_op = biases['bd1'].assign(bd1)
        sess.run(assign_op)
        out_b = data['h']
        assign_op = biases['out'].assign(out_b)
        sess.run(assign_op)
        #print sess.run(tf.argmax(pred, 1), feed_dict={x: testX, keep_prob: 1.})
        return sess.run(tf.argmax(pred, 1), feed_dict={x: testX, keep_prob: 1.})
