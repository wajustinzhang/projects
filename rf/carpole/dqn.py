import gym
import tensorflow as tf
import numpy as np

class dqn:
    def __index__(self, env):
        self.env = gym.make(env)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.HIDDEN1_SIZE = 20
        self.HIDDEN2_SIZE = 20
        self.LAMBDA = 0.1
        self.LEARNING_RATE = 0.5

    def init_network(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        with tf.name_space('hidden1'):
            W1 = tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE]), stddev=0.01, name='W1')
            b1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name='b1')
            h1 = tf.nn.tanh(tf.matmul(self.x, W1) + b1)

        with tf.name_space('hidden2'):
            W2 = tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE]), stddev=0.01, name='W2')
            b2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name='b2')
            h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

        with tf.name_space('output'):
            W3 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size]), stddev=0.01, name='W3')
            b3 = tf.Variable(tf.zeros(self.HIDDEN3_SIZE), name='b3')
            self.Q = tf.matmul(h2,W3) + b3

        # loss
        self.targetQ = tf.placeholder(tf.float32, [None])
        self.targetActionMast = tf.placeholder(tf.float32, [None, self.output_size])
        q_values = tf.reduce_sum(tf.mul(self.Q, self.targetActionMast), reduction_indices=[1])
        self.loss = tf.reduce_mean(tf.square(tf.sub(q_values, self.targetQ)))
        self.weights = [W1,b1,W2, b2, W3, b3]

        # regulization
        for w in [W1, W2, W3]:
            self.loss += self.LAMBDA * tf.reduce_sum(tf.square(w))

        # Training
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)


    def training(self, num_episodes=2000):
        session = tf.Session()
        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter('./log', self.session.graph)

        session.run(tf.initialzie_all_variables())
        total_steps = 0
        step_counts = []
        target_weights = session.run(self.weights)
        for episode in range(num_episodes):
            state = self.env.reset()
            steps = 0

            for step in range(self.MAX_STEPS):
                pass

