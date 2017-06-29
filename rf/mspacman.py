import gym

env=gym.make('CartPole-v0')
#env=gym.make('MountainCar-v0')
#env = gym.make('MsPacman-v0')  #
#env = gym.make('Hopper-v1')
env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('is done? {}'.format(done))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break