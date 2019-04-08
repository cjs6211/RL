import tensorflow as tf
import gym
import numpy as np

env = gym.make("CartPole-v0")

#input and output size based on the env
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1

#Feed-forward part
X = tf.placeholder(shape = [1, input_size], dtype=tf.float32, name="input_x") #state input
W1 = tf.get_variable(name="W1", shape=[input_size, output_size],
                    initializer=tf.contrib.layers.xavier_initializer())
Qpred = tf.matmul(X,W1)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Qpred-Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#init Q-table
num_episodes = 2000
#discount factor
dis = 0.9
rList = []

#init tf session
init = tf.global_variables_initializer()

#open with tensorflow session
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        e = 1.0/((i//10)+10)
        rAll = 0
        step_count = 0
        done = False

        while not done:
            step_count += 1
            x = np.reshape(s, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: x})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, info = env.step(a)
            if done:
                #Update Q, when it's terminal state
                Qs[0, a] = -100
            else:
                x1 = np.reshape(s1, [1, input_size])
                #Update Q when it's not terminal state
                Qs1 = sess.run(Qpred, feed_dict={X: x1})
                Qs[0, a] = reward + dis * np.max(Qs1)

            #Train the network using Y and predicted Q values
            sess.run(train, feed_dict={X:x, Y:Qs})
            s = s1
        rList.append(step_count)
        print("episode:{}, step_count:{}".format(i + 1, step_count ))

        if len(rList)>10 and np.mean(rList[-10:])>500:
            break

    #evaluation
    observation = env.reset()
    reward_sum =0
    while True:
        env.render()

        x= np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X:x})
        a = np.argmax(Qs)

        observation, reward, done, info = env.step(a)
        reward_sum += reward
        if done:
            print("Total score:{}".format(reward_sum))
            break
