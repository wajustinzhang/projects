import tensorflow as tf

# hypothesis
w = tf.Variable([.3, 0.4,0.5, .3], dtype=tf.float32)
b=tf.Variable([-.3])
x=tf.placeholder(tf.float32)
linear_model = w*x + b

# loss function
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))

# define an optimizer with learning rate 0.01
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# create a session object
sess = tf.Session()

# run initialization
init = tf.global_variables_initializer()
sess.run(init)

x_train = [1,2,3,4]
y_label = [0,-1,-2,-3]
for i in range(2000):
    sess.run(train, {x:x_train, y: y_label})
    if i %100 == 0:
        print('w is {},  b is {}'.format(sess.run(w), sess.run(b)))

curr_W, curr_b, curr_loss = sess.run([w, b, loss], {x:x_train, y:y_label})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

