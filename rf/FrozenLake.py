import gym
import numpy as np

from gym import wrappers

env = gym.make("FrozenLake-v0")
#env = wrappers.Monitor(env, "./gym-results")

Q = np.zeros([env.observation_space.n, env.action_space.n])
lr = .8
y = .95

num_episodes = 2000
rList = []

for i in range(num_episodes):
    s = env.reset()

    rAll = 0
    d = False
    j=0
    while j<99:
        j+=1
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        print('action is {}'.format(a))
        s1, r, d, _ = env.step(a)
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1

        if d == True:
            break

print ("score over time: {}".format(sum(rList)/num_episodes))
print('Final Q-Table Values {}'.format(Q))