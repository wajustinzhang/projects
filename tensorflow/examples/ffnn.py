import math
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


IMAGE_PIXELS = 784

# Input
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Hidden 1
w1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, 128], stddev=1.0/math.sqrt(float(IMAGE_PIXELS))))
b1 = tf.Variable(tf.truncated_normal([128], stddev=1.0/math.sqrt(float(IMAGE_PIXELS))))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1) # 128

# Hidden 2
w2 = tf.Variable(tf.truncated_normal([128, 32], stddev=1.0/math.sqrt(float(128))))
b2 = tf.Variable(tf.truncated_normal([32], stddev=1.0/math.sqrt(float(128))))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

# output layer
w = tf.Variable(tf.truncated_normal([32, 10], stddev=1.0/math.sqrt(float(10))))
b = tf.Variable(tf.truncated_normal([10], stddev=1.0/math.sqrt(float(10))))
y = tf.matmul(h2, w) + b

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=tf.Variable(0, trainable= False))

# Train
saver = tf.train.Saver()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

log_dir = './log'
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

with tf.Session() as sess:
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./log', sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        batch = mnist.train.next_batch(50)
        feed_dict = {x: batch[0], y_: batch[1]}
        _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        if step % 100 == 0:
            checkpoint_file = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (step, train_accuracy))
            print('loss value:{}'.format(loss_value))

    # Evaluate
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
