from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def biase_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_poll_2X2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# First convlayer
W_conv1 = weight_var([5,5,1,32])
b_conv1 = biase_var([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  #  28 X 28 X 32
h_pool1 = max_poll_2X2(h_conv1)   #  14 X 14 X 32

# Second layer
W_conv2 = weight_var([5,5,32,64])
b_conv2 = biase_var([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #  14 X 14 X 64
# if remove pool1
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) #   28 X 28 X 64

h_pool2 = max_poll_2X2(h_conv2)  # 7 X 7 X 64

# fully connected layer
# W_fc1 = weight_var([7*7*64, 1024])
W_fc1 = weight_var([28*28*64, 1024])
b_fc1 = biase_var([1024])

#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_pool2_flat = tf.reshape(h_conv2, [-1, 28*28*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer
W_fc2 = weight_var([1024,10])
b_fc2 = biase_var([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))