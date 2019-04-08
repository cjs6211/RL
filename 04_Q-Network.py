import tensorflow as tf
import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def one_hot(x):
    return np.identity(16)[x:x+1]

env = gym.make("FrozenLake-v0")

#input and output size based on the env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

#Feed-forward part
X = tf.placeholder(shape = [1, input_size], dtype=tf.float32) #state input
W = tf.Variable(tf.random_uniform(shape=[input_size, output_size], minval=0.0, maxval=0.01))
Qpred = tf.matmul(X,W)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Qpred-Y))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#init Q-table
num_episodes = 2000
#discount factor
dis = 0.99
rList = []

#init tf session
init = tf.global_variables_initializer()

#open with tensorflow session
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        e = 1.0/((i//50)+10)
        rAll = 0
        done = False
        local_loss = 0
        total_loss = 0

        while not done:
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, info = env.step(a)
            if done:
                #Update Q, when it's terminal state
                Qs[0, a] = reward
            else:
                #Update Q when it's not terminal state
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                Qs[0, a] = reward + dis * np.max(Qs1)

            #Train the network using Y and predicted Q values
            local_loss,_ = sess.run([loss, train], feed_dict={X:one_hot(s), Y:Qs})

            total_loss += local_loss
            rAll += reward
            s = s1
        rList.append(rAll)
        if i % 100 == 99:
            print("episode:{}, avg_train_loss:{}".format( i + 1, total_loss/(i+1) ) )

    print("Success rate: " +str(sum(rList)/num_episodes))
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show(block = True)
    plt.interactive(False)

