import sys
import gym
import pylab
import numpy as np
import tensorflow as tf

EPISODES = 5000

#Advantage Actor Critic agent
class A2CAgent:
    def __init__(self, state_size, action_size, sess):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.sess = sess

        #hyperparameters
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.01

        #generate policy and Q network
        self.state_in = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
        self.build_actor()
        self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

        #load saved model
        if self.load_model:
            self.actor.load_weight("./save_model/carpole_actor_trained.h5")
            self.critic.load_weight("./save_model/carpole_ctritic_trained.h5")

    """policy network"""
    def build_actor(self):
        W1 = tf.get_variable(name="policy_W1", shape=[self.state_size, 24],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.state_in, W1))
        W2 = tf.get_variable(name="policy_W2", shape=[24, self.action_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(layer1, W2)
        self.action_output = tf.nn.softmax(logits=logits)

    """Q network"""
    def build_critic(self):
        W1 = tf.get_variable(name="critic_W1", shape=[self.state_size, 24],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.state_in, W1))
        W2 = tf.get_variable(name="critic_W2", shape=[24, 24],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer2 = tf.nn.relu(tf.matmul(layer1, W2))
        W3 = tf.get_variable(name="critic_W3", shape=[24, self.value_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        self.Qpred = tf.matmul(layer2, W3)

    def get_action(self, state):
        policy = sess.run(self.action_output, feed_dict={self.state_in: state})
        action = np.random.choice(policy[0], p=policy[0])
        action = np.argmax(policy == action)
        return action

    def actor_optimizer(self):
        self.chosen_action = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.advantage = tf.placeholder(shape=[None, ], dtype=tf.float32)

        action_prob = tf.reduce_sum(self.chosen_action*self.action_output, axis=1)
        self.policy_loss = -tf.reduce_sum(tf.log(action_prob)*self.advantage)

        train = tf.train.AdamOptimizer(learning_rate=self.actor_lr).minimize(self.policy_loss)
        return train

    def critic_optimizer(self):
        self.Qs = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.Qs - self.Qpred))
        train = tf.train.AdamOptimizer(learning_rate=self.critic_lr).minimize(self.critic_loss)
        return train

    def train_model(self, state, action, reward, next_state, done):
        value = sess.run(self.Qpred, feed_dict={self.state_in:state})
        next_value = sess.run(self.Qpred, feed_dict={self.state_in:next_state})

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value[0]
            target = [reward]
        else:
            advantage = reward + self.discount_factor*next_value[0] - value[0]
            target = reward + self.discount_factor*next_value[0]

        l1, _ = sess.run([self.policy_loss, self.actor_updater], feed_dict={self.state_in:state,
                                                self.chosen_action:act,
                                                self.advantage:advantage})
        l2, _ = sess.run([self.critic_loss, self.critic_updater], feed_dict={self.state_in:state,
                                                 self.Qs:target})
        #print("policy loss:{}, critic loss:{}".format(l1, l2))

if __name__=="__main__":
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 10001
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    with tf.Session() as sess:
        agent = A2CAgent(state_size=state_size, action_size=action_size, sess=sess)
        tf.global_variables_initializer().run()
        step_counts, episodes = [], []

        for e in range(EPISODES):
            done = False
            step_count = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            while not done:
                step_count += 1
                if agent.render:
                    env.render()

                action = agent.get_action(state=state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                if done:
                    reward = -100
                    step_counts.append(step_count)
                    episodes.append(e)
                    pylab.plot(episodes, step_counts, 'b')
                    pylab.savefig("./save_graph/cartpole_a2c.png")
                    print("episode: {} steps: {}".format(e, step_count))

                agent.train_model(state, action, reward, next_state, done)
                if step_count>10000:
                    break

                state = next_state