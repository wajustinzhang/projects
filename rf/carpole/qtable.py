import gym
import numpy as np

class QtableLearn:
    def __init__(self, states_nums, action_nums, alpha, gamma, explore_rate, explore_decay_rate):
        self.states_nums = states_nums
        self.action_nums = action_nums
        self.alpha = alpha
        self.gamma = gamma
        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_decay_rate
        self.qtable = np.random.uniform(low=-1, high=1, size=(states_nums, action_nums))

    def init_state(self, init_state):
        self.current_state = init_state
        self.current_action = self.qtable[init_state].argsort()[-1]
        return self.current_action

    def move(self, state, reward):
        explore = (1-self.explore_rate) <= np.random.uniform(0,1)

        # select action
        if explore:
            action = np.random.randint(0, self.action_nums - 1)
        else:
            action = self.qtable[state].argsort()[-1]

        self.explore_rate *= self.explore_rate * self.explore_rate_decay

        self.qtable[self.current_state, self.current_action] = \
            (1 - self.alpha) * self.qtable[state, action] + \
            self.alpha * (reward + self.gamma * self.qtable[state, action])

        self.current_state = state
        self.current_action = action

        return action

    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(np.abs(feature))), features)))

    def getQtable(self):
        return self.qtable

if __name__ == '__main__':
    np.random.seed(0)
    env = gym.make('CartPole-v0')
    # env.monitor.start('./cartpole-experiment-v0', force=True)

    feature_nums = env.observation_space.shape[0]
    action_nums = env.action_space.n
    learner = QtableLearn(states_nums=10**feature_nums, action_nums=action_nums,
                          alpha=0.2, gamma=1, explore_rate= 0.5, explore_decay_rate= 0.99)

    goal_avg_steps = 195
    max_steps_per_episode = 200
    last_time_steps = np.array(0)
    for i in range(5000):
        observation = env.reset()
        action = learner.init_state(learner.build_state(observation))

        for step in range(max_steps_per_episode - 1):
            observation, reward, done, info = env.step(action)

            if done:
                last_time_steps = np.append(last_time_steps, step + 1)
                break

            action = learner.move(learner.build_state(observation), reward)

    print('average steps to reach goal: {}'.format(last_time_steps.mean()))
    print('final qtable {}'.format(learner.getQtable()))

    #env.monitor.close()

