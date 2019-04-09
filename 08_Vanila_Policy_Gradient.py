import tensorflow as tf
import numpy as np
import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

try:
    xrange = xrange
except:
    xrange = range

env = gym.make("CartPole-v0")
env._max_episode_steps = 10001

gamma = 0.99

def discount_reward(r):
    #take 1d float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        #feed forward network
        self._state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        W1 = tf.get_variable(name="W1", shape=[s_size, h_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self._state_in, W1))
        W2 = tf.get_variable(name="W2", shape=[h_size, a_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(layer1, W2)
        self._output = tf.nn.softmax(logits=logits)
        self._chosen_action = tf.argmax(self._output, 1)

        #training procedure.
        #we feed the reward and chosen action into the network to compute the loss and use it update the network
        self._reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self._action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self._indices = tf.range(0, tf.shape(self._output)[0]) * tf.shape(self._output)[1] + self._action_holder
        self._responsible_outputs = tf.gather(tf.reshape(self._output, [-1]), self._indices)

        self.loss = -tf.reduce_mean(tf.log(self._responsible_outputs)*self._reward_holder)

        vars = tf.trainable_variables()
        self._gradient_holders = []
        for idx, var in enumerate(vars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self._gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, vars)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self._gradient_holders, vars))

tf.reset_default_graph()
myAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=20)
total_episodes = 5000
update_frequency = 5

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(gradBuffer):
        gradBuffer[idx]  = grad * 0

    for i in range(total_episodes):
        state = env.reset()
        done = False
        step_count = 0
        running_reward = 0
        ep_history = []
        while not done:
            env.render()
            #stochastic pick an action given network output
            a_dist = sess.run(myAgent._output, feed_dict={myAgent._state_in:[state]})
            action = np.random.choice(a_dist[0], p=a_dist[0])
            action = np.argmax(a_dist==action)

            next_state, reward, done, info = env.step(action)
            ep_history.append([state, action, reward, next_state])
            state = next_state
            running_reward += reward
            step_count +=1

            if done:
                #update network
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_reward(ep_history[:, 2])
                feed_dict = {myAgent._reward_holder:ep_history[:, 2],
                             myAgent._action_holder:ep_history[:, 1],
                             myAgent._state_in:np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)

                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i%update_frequency == 0 and i != 0:
                    feed_dict = dict(zip(myAgent._gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)

                    for idx, grad in enumerate(gradBuffer):
                        gradBuffer[idx] = grad * 0

            if step_count>10000:
                break

        #total_reward.append(running_reward)
        #total_length.append(step_count)
        print("Episode: {} Steps: {}".format(i, step_count))


