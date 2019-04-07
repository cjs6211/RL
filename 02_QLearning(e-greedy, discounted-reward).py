import tensorflow as tf
import gym
import numpy as np
from gym.envs.registration import register
import matplotlib.pyplot as plt
import random as pr

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery':False}
)

env = gym.make("FrozenLake-v3")

#init Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
#discount factor
dis = 0.99
rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    e = 1./((i//100)+1)

    while not done:
        if np.random.rand(1) < e :
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)
        #update Q-Table
        Q[state, action] = reward + dis*np.max(Q[new_state, :])

        rAll += reward
        state = new_state

        #env.render()
        #print("state:", state, "action:", action, "reward:", reward, "info:", info)

    if done:
        rList.append(rAll)
        #print("Finished with reward:", reward)

print("Success rate: " +str(sum(rList)/num_episodes))
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
