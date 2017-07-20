import gym
env = gym.make('CartPole-v0')
print(env.ac)
for i_episode in range(20):
    print('In {}th episode'.format(i_episode))
    observation = env.reset()
    for t in range(15):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('observation {}, reward{}, info{}'.format(observation, reward, info))

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break