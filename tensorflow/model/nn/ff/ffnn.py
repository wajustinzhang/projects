import config
import tensorflow as tf
import math

class ffnn:
    def __init(self, configFileName):
        self.cfg = config(configFileName).readConfig()

    def getInstance(self, inputdata, inputsize, outputsize):
        layers = self.cfg['layers'] # it is a tuple
        pairs = [(layers[i], layers[i+1]) for i in range(len(layers)-1)]

        # config input
        weights = tf.Variable(tf.truncated_normal(inputsize, pairs[0][0], stddev=1.0/math.sqrt(float(pairs[0][0]))), name='weights')
        biases = tf.variable(tf.zeros([pairs[0][0]]), name='biases')
        layerInput = tf.nn.relu(tf.matmul(inputdata, weights) + biases)

        # config hidden layers
        for pair in pairs:
            weights = tf.Variable(tf.truncated_normal(pair[0], pair[1], stddev=1.0/math.sqrt(float(pair[0]))), name='weights')
            biases = tf.variable(tf.zeros([pair[1]]), name='biases')
            layerInput = tf.nn.relu(tf.matmul(layerInput, weights) + biases)

        # config output
        weights = tf.Variable(tf.truncated_normal([layerInput, outputsize],stddev=1.0 / math.sqrt(float(pairs(len(layers)-1))[1])),name='weights')
        biases = tf.Variable(tf.zeros([outputsize]),name='biases')
        self.model = tf.matmul(layerInput, weights) + biases

        return self

    def define_loss(self, labels):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.model, name='xentropy')
        self.loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return self

    def start_training(self):
        learning_rate = self.cfg['learning_rate']
        tf.summary.scalar('loss', self.loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op=optimizer.minimize(self.loss, global_step=global_step)
        return self

    def evaluation(self, labels):
        correct = tf.nn.in_top_k(self.model, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.to_int32))
